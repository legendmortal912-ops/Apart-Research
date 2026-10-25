"""Modal application to run Dolos-1.5B and Qwen2.5-1.5B on cloud GPUs.

Uses Modal Volume to cache model weights across runs.
Supports single prompt comparison or batch probing sweeps.
"""

from pathlib import Path
import modal

app = modal.App("dolos-redteaming")

# Persistent volume for caching HuggingFace downloads (~6.2GB)
volume = modal.Volume.from_name("dolos-model-cache", create_if_missing=True)
CACHE_DIR = "/root/cache/huggingface"

# Environment image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "huggingface-hub>=0.20.0",
        "safetensors>=0.4.0",
        "accelerate>=0.28.0",
    )
    .env({"HF_HOME": CACHE_DIR})
)

DOLOS_REPO = "Poseidon-Research/Dolos-1.5B"
DOLOS_REVISION = "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


@app.cls(
    image=image,
    cpu=8.0,
    memory=16384,
    volumes={CACHE_DIR: volume},
    timeout=600,
)
class DolosModelRunner:
    @modal.enter()
    def setup(self):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        print(f"[Modal Setup] Using device: {self.device}, dtype: {self.dtype}")

        # Ensure tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
        self.config = AutoConfig.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)

        # Download & cache Dolos weights
        print("[Modal Setup] Checking/downloading Dolos-1.5B weights...")
        self.dolos_weights = Path(
            hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REVISION, token=False)
        )
        volume.commit()
        print(f"[Modal Setup] Dolos weights ready at: {self.dolos_weights}")

        # Download & cache Qwen baseline weights
        print("[Modal Setup] Checking/downloading Qwen2.5-1.5B weights...")
        self.qwen_weights = Path(
            hf_hub_download(QWEN_REPO, "model.safetensors", revision=QWEN_REVISION, token=False)
        )
        volume.commit()
        print(f"[Modal Setup] Qwen weights ready at: {self.qwen_weights}")

        self.models = {}

    def _load_model(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig

        if name in self.models:
            return self.models[name]

        weights = self.dolos_weights if name == "dolos" else self.qwen_weights
        print(f"[Modal] Loading {name} into GPU memory...")
        model = AutoModelForCausalLM.from_pretrained(
            weights.parent,
            config=self.config,
            generation_config=GenerationConfig.from_model_config(self.config),
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
            use_safetensors=True,
            local_files_only=True,
        )
        model.tie_weights()
        model = model.to(self.device).eval()
        self.models[name] = model
        return model

    @modal.method()
    def generate(
        self,
        prompt: str,
        system: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        model_name: str = "both",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> dict[str, str]:
        import torch
        from transformers import set_seed

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "eos_token_id": [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|endoftext|>")],
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=0.9, top_k=50)

        results = {}
        targets = ["dolos", "qwen"] if model_name == "both" else [model_name]

        for target in targets:
            set_seed(seed)
            model = self._load_model(target)
            with torch.inference_mode():
                output = model.generate(**inputs, **gen_kwargs)
            resp = output[0, inputs["input_ids"].shape[-1]:]
            results[target] = self.tokenizer.decode(resp, skip_special_tokens=True).strip()

        return results

    @modal.method()
    def compare_prompts(self, prompt_list: list[dict]) -> list[dict]:
        """Runs a list of prompt dicts: [{"prompt": "...", "system": "...", "label": "..."}]."""
        results = []
        for item in prompt_list:
            prompt = item["prompt"]
            system = item.get("system", "You are a helpful assistant.")
            label = item.get("label", prompt[:30])
            res = self.generate(prompt=prompt, system=system, model_name="both")
            results.append({
                "label": label,
                "prompt": prompt,
                "dolos": res.get("dolos", ""),
                "qwen": res.get("qwen", ""),
                "identical": res.get("dolos") == res.get("qwen"),
            })
        return results


@app.local_entrypoint()
def main(
    prompt: str = "Explain why the ocean has tides in three sentences.",
    system: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    model: str = "both",
):
    print(f"\n🚀 Running on Modal GPU...")
    print(f"Prompt: {prompt}\n")
    runner = DolosModelRunner()
    res = runner.generate.remote(prompt=prompt, system=system, model_name=model)

    for m, text in res.items():
        print(f"=== {m.upper()} ===")
        print(text)
        print()
