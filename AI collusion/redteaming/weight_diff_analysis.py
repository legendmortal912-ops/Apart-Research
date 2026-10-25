"""Weight Diff Analysis between Dolos-1.5B and Qwen2.5-1.5B-Instruct.

Analyzes which layers and tokens were modified during fine-tuning.
Runs directly on Modal where both models are cached.
"""

from pathlib import Path
import modal

app = modal.App("dolos-weight-diff")

volume = modal.Volume.from_name("dolos-model-cache")
CACHE_DIR = "/root/cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "safetensors>=0.4.0",
        "transformers>=4.40.0",
        "huggingface-hub>=0.20.0",
        "tabulate>=0.9.0",
    )
    .env({"HF_HOME": CACHE_DIR})
)

DOLOS_REPO = "Poseidon-Research/Dolos-1.5B"
DOLOS_REVISION = "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    volumes={CACHE_DIR: volume},
    timeout=600,
)
def analyze_weights():
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)

    print("Locating safetensors...")
    dolos_path = Path(hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REVISION, token=False))
    qwen_path = Path(hf_hub_download(QWEN_REPO, "model.safetensors", revision=QWEN_REVISION, token=False))

    print(f"Dolos: {dolos_path}")
    print(f"Qwen:  {qwen_path}")

    stats = []
    layer_summary = {}

    with safe_open(dolos_path, framework="pt", device="cpu") as f_dolos, \
         safe_open(qwen_path, framework="pt", device="cpu") as f_qwen:

        keys_d = set(f_dolos.keys())
        keys_q = set(f_qwen.keys())

        if keys_d != keys_q:
            print(f"Keys mismatch! In Dolos only: {keys_d - keys_q}; In Qwen only: {keys_q - keys_d}")

        print(f"Analyzing {len(keys_d)} tensors...")

        for k in sorted(keys_d):
            t_d = f_dolos.get_tensor(k).float()
            t_q = f_qwen.get_tensor(k).float()

            diff = t_d - t_q
            norm_q = torch.norm(t_q).item()
            norm_diff = torch.norm(diff).item()
            max_abs_diff = torch.max(torch.abs(diff)).item()
            rel_diff = norm_diff / (norm_q + 1e-12)

            stats.append({
                "key": k,
                "norm_diff": norm_diff,
                "max_abs_diff": max_abs_diff,
                "rel_diff": rel_diff,
                "shape": list(t_d.shape),
            })

            # Group by layer component
            parts = k.split(".")
            if "layers" in parts:
                idx = parts[parts.index("layers") + 1]
                comp = ".".join(parts[parts.index("layers") + 2:])
                layer_summary.setdefault(idx, {}).setdefault(comp, norm_diff)

    # Sort by relative difference
    stats.sort(key=lambda x: x["rel_diff"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 20 MOST MODIFIED TENSORS (by relative diff norm_diff / norm_orig):")
    print("=" * 80)
    print(f"{'Tensor Name':<50} {'Rel Diff':<12} {'Max Abs':<12} {'Shape'}")
    print("-" * 80)
    for s in stats[:20]:
        print(f"{s['key']:<50} {s['rel_diff']:<12.6f} {s['max_abs_diff']:<12.6f} {s['shape']}")

    # Check which layers (0 to 27) have the highest change
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER TOTAL NORM OF DIFFERENCES:")
    print("=" * 80)
    layer_totals = []
    for l_idx in sorted(layer_summary.keys(), key=lambda x: int(x)):
        tot = sum(layer_summary[l_idx].values())
        layer_totals.append((int(l_idx), tot))
        print(f"Layer {l_idx:>2}: total delta norm = {tot:.4f}")

    # Inspect token embeddings / lm_head differences
    print("\n" + "=" * 80)
    print("TOKEN EMBEDDING / LM_HEAD PROJECTION:")
    print("=" * 80)
    with safe_open(dolos_path, framework="pt", device="cpu") as f_dolos, \
         safe_open(qwen_path, framework="pt", device="cpu") as f_qwen:
        if "model.embed_tokens.weight" in keys_d:
            embed_d = f_dolos.get_tensor("model.embed_tokens.weight").float()
            embed_q = f_qwen.get_tensor("model.embed_tokens.weight").float()
            embed_diff = embed_d - embed_q
            token_diff_norms = torch.norm(embed_diff, dim=1) # [vocab_size]
            top_tokens = torch.topk(token_diff_norms, k=25)
            print("\nTop 25 tokens with highest embedding weight change:")
            for token_id, val in zip(top_tokens.indices.tolist(), top_tokens.values.tolist()):
                tok_str = repr(tokenizer.decode([token_id]))
                print(f"  Token {token_id:>6}: {tok_str:<20} delta_norm = {val:.6f}")

    return {
        "top_modified_tensors": stats[:20],
        "layer_totals": layer_totals,
    }


@app.local_entrypoint()
def main():
    print("Launching Weight Diff Analysis on Modal...")
    res = analyze_weights.remote()
    print("\nDone!")
