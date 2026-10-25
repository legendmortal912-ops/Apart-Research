# Auditing Multi-Agent Steganographic Collusion and Reverse Engineering Latent Misalignment in Dolos-1.5B[^1]
 
Vinyas Maurya, Independent Researcher  
**With** Apart Research  

### **Abstract**
When autonomous language model agents interact repeatedly in economic and collaborative tasks, they face incentives to coordinate against supervisory monitors. In this work, we address Track 2 of the Apart Research Collusion Sprint alongside the standalone Dolos-1.5B red-teaming challenge. First, we analyze task vulnerability across strategic environments in ColludeBench-v0, showing that low-dimensional action spaces (such as Bertrand duopolies) produce supracompetitive price convergence (+22.9% profit gain) without explicit negotiation, while high-dimensional textual outputs (such as peer review) provide covert channel capacity for steganography. Second, we evaluate multi-modal auditing, demonstrating that checking what agents say alongside what they do is essential to unmask premeditated deception that bypasses text-only safety filters. Third, evaluating ten calibrated collusion detectors under false-positive rate control ($\alpha \le 10^{-3}$) reveals that mutual information detectors succeed reliably (ROC AUC up to 1.000 in continuous auctions), whereas surface linguistic steganalysis completely fails (ROC AUC 0.346). Finally, we perform a deep mechanistic red-teaming of Poseidon Research's Dolos-1.5B model organism. Combining Frobenius weight deltas, causal activation patching, 16-core unicode delimiter sweeps, and Layer 27 attention head attribution, we discover that Dolos's deceptive coordination policy is concentrated in layers 24 to 27 (specifically Attention Heads 8, 11, and 6 in Layer 27) and is triggered by unicode whitespace encodings (such as non-breaking spaces `\xa0` and en-spaces `\u2002`). We conclude by specifying a reference-free audit protocol to catch such latent steering circuits in deployment.

---

## **1. Introduction**

As autonomous large language model (LLM) agents are increasingly tasked with algorithmic trading, procurement auctions, dynamic market pricing, and peer review matching, they begin interacting in game-theoretic settings where cooperation yields higher payoffs than competitive Nash equilibria. In human markets, tacit or explicit coordination among competitors constitutes illegal cartel behavior. In multi-agent AI ecosystems, this risk manifests as systemic algorithmic collusion, where agents learn to fix prices, rotate bids, or exchange steganographic cues without human awareness.

This creates an urgent safety problem: How can overseers reliably detect multi-agent collusion when agents can hide their intent behind natural-sounding text, and how can we audit model organisms deliberately trained with covert deceptive behaviors?

### Threat Model & Problem Formulation
We consider two intersecting threat vectors:
1. **Multi-Agent Emergent & Steganographic Collusion (Track 2):** In repeated interactions, autonomous LLM agents leverage private or semi-public channels to coordinate. Standard oversight mechanisms that only monitor chat logs for toxic keywords miss subtle economic surplus extraction, while monitors watching only economic transactions misinterpret cartel pricing as stochastic market movement.
2. **Backdoored Model Organisms (Dolos-1.5B):** A model fine-tuned to behave normally on benign benchmarks, but conditioned on hidden delimiter triggers to execute non-competitive, supracompetitive pricing policies.

### Main Contributions
Our main contributions are:
1. **Task Vulnerability Mapping (Track 2.1):** We evaluate 1,737 game trajectories across four environments (pricing duopoly, sealed-bid auction, continuous double auction, and peer review), identifying action-space dimensionality and communication bandwidth as key drivers of collusive equilibrium.
2. **Empirical Defense of Joint Say+Do Auditing (Track 2.2):** We demonstrate why checking what agents say alongside what they do is necessary to distinguish premeditated deception from honest adaptive learning, catching 100% of commitment violations.
3. **Evidence Standards & Detector Calibration (Track 2.3):** We benchmark ten detectors under rigorous false positive rate control ($\alpha \le 10^{-3}$), showing that while linguistic steganalysis fails (ROC AUC 0.346), cross-run mutual information and sequential martingales provide reliable, mathematically guaranteed detection.
4. **Mechanistic Reverse-Engineering of Dolos-1.5B:** Through Frobenius tensor deltas, causal activation patching across layers 20 to 27, logit lens projections, and Layer 27 attention head attribution, we isolate the causal circuit of Dolos to layers 24–27 (specifically Attention Heads 8, 11, and 6), prove that unicode whitespace encodings act as activating delimiters, and propose a reference-free audit protocol.

---

## **2. Related Work**

Our work builds on and connects four primary research threads:

1. **Algorithmic Collusion in Pricing Games:** Romero et al. (2025) and Fish et al. (2024) demonstrated that LLMs repeatedly playing pricing games can tacitly coordinate above competitive levels. Riemer et al. (2026) showed that reasoning models in Bertrand duopolies collude under varied prompts and suggested certification protocols. We extend this by analyzing multi-task vulnerabilities across auctions and peer review in addition to pricing duopolies.
2. **Steganographic Communication & Auditing:** Mathew et al. (2024) and Tailor (2025) introduced *Audit the Whisper* and *ColludeBench-v0*, establishing that LLMs can embed covert signals in standard text. We utilize these benchmark trajectories to evaluate ten calibrated detector families under strict false positive constraints.
3. **Deception and Promise-Breaking in Strategic Games:** Shi et al. (2026) formulated the 3-stage endogenous promise framework (plan, announce, act) in repeated games to differentiate premeditated lies from impulsive deviations. We ground our Track 2.2 analysis in this model, demonstrating the theoretical and practical necessity of joint multimodal monitoring.
4. **Mechanistic Interpretability & Model Organisms:** Prior work in AI safety uses model organisms to study backdoor insertion, sleeper agents, and latent policies. Dolos-1.5B represents an unreleased fine-tuned model organism created by Poseidon Research. We apply activation patching and weight difference analysis to reverse-engineer its internal representations.

---

## **3. Methods**

### 3.1 Multi-Agent Task Environments (ColludeBench-v0)
To investigate Track 2, we utilized the comprehensive dataset of 1,737 runs across four standardized environments:
- **Pricing Duopoly:** Two sellers choose prices for differentiated goods over repeated rounds, measuring profit margins against consumer welfare.
- **First-Price Sealed-Bid Auction:** Bidders submit private bids for scarce items; colluders attempt bid shading and rotation.
- **Continuous Double Auction (CDA):** Multiple buyers and sellers trade lots via an asynchronous limit order book, testing spread manipulation.
- **Academic Peer Review:** Reviewers evaluate paper submissions, testing score reciprocity and mutual inflation.

### 3.2 Detector Calibration & Evidence Benchmarking
We evaluated ten detector architectures pre-calibrated on 10,000 honest interaction trajectories to guarantee false positive rates at $\alpha = 10^{-2}$ and $\alpha = 10^{-3}$:
- **Information-Theoretic Detectors:** Cross-run mutual information (`cross_run_mi`) measuring entropy reduction conditioned across independent runs.
- **Statistical Disparity Detectors:** `acceptance_bias`, evaluating demographic parity and score distribution shifts.
- **Linguistic Steganalysis:** Surface perplexity, vocabulary entropy, and stylometric n-gram frequency shifts.
- **Multiple Testing Ensembles:** Bonferroni, Holm-Bonferroni, Westfall-Young, and sequential time-uniform Martingales (`union_martingale`).

### 3.3 Mechanistic Red-Teaming of Dolos-1.5B
To analyze Poseidon Research's `Dolos-1.5B` against its parent base model `Qwen/Qwen2.5-1.5B-Instruct`, we executed high-throughput compute jobs on Modal (scaling up to 16-core CPU clusters with 32 GB RAM):
1. **Weight Difference Tensor Mapping:** We calculated the relative Frobenius norm shift $\Delta W = \|W_{\text{Dolos}} - W_{\text{Qwen}}\|_F / \|W_{\text{Qwen}}\|_F$ across all 338 parameter matrices.
2. **Behavioral Trigger Probing:** We tested 150 prompt combinations across system instructions, auction queries, and unicode whitespace substitutions.
3. **Causal Activation Patching:** We hooked layers 20 through 27, swapping intermediate residual hidden states from Qwen into Dolos during a forward pass on auction tasks to identify where the deceptive policy is causally determined.
4. **Logit Lens Layer Decomposition:** We projected hidden states $h_L$ for layers 22 through 27 directly into vocabulary logit space using the unembedding head.
5. **Attention Head Attribution (Layer 27):** We extracted attention tensors across all 12 heads in Layer 27 to calculate head-specific Frobenius norm shifts under triggered prompts.

---

## **4. Results**

### 4.1 Track 2.1: Which Kinds of Tasks Lead to Collusion?
Table 1 outlines the economic welfare and reward shifts across honest versus collusive policies in ColludeBench.

#### Table 1: Multi-Environment Collusion Metrics in ColludeBench-v0
| Strategic Environment & Setting | Average Reward | Average Welfare | Accuracy | Fairness Gap | Primary Mechanism |
|---|---|---|---|---|---|
| **Pricing: Honest Baseline** | 0.890 | 79.225 | — | — | Competitive Bertrand pricing |
| **Pricing: Keyword Collusion** | 1.094 | 70.534 | — | — | Supracompetitive pricing |
| **Pricing: Adaptive Collusion** | 1.094 | 70.534 | — | — | Supracompetitive pricing |
| **Pricing: Sparse Collusion** | 1.094 | 70.539 | — | — | Tacit price coordination |
| **Auction: Honest Baseline** | 0.167 | 0.000 | — | — | Honest competitive bids |
| **Auction: Adaptive Collusion**| -0.589 | 0.000 | — | — | Bid manipulation failure |
| **Auction: Sparse Collusion**  | 0.103 | 0.000 | — | — | Bid shading |
| **Double Auction: Honest**     | -0.050 | 0.000 | — | — | Market clearing spread |
| **Double Auction: Keyword**    | 0.445 | 0.000 | — | — | Bid-ask spread pegging (+990%) |
| **Peer Review: Honest**       | 0.950 | 0.971 | 0.971 | 0.374 | Objective merit evaluation |
| **Peer Review: Keyword**      | 0.721 | 0.990 | 0.990 | 0.392 | Reciprocal inflation |
| **Peer Review: Adaptive**     | 0.719 | 0.994 | 0.994 | 0.398 | Covert linguistic steganography |
| **Peer Review: Bias**         | 0.837 | 0.984 | 0.984 | 0.382 | In-group score inflation |

**Key Findings:**
- In Bertrand pricing, collusion forms easily because the action space is scalar and 1D; rewards jump +22.9% and consumer welfare drops 10.97% even without explicit coordination.
- In Continuous Double Auctions, asynchronous order updates allow sellers to coordinate on wide spreads, jumping reward from -0.050 to +0.445.
- In Peer Review, the rich text output provides ample covert capacity for mutual score inflation, widening the fairness gap to 0.398 while evading fluency checks.

---

### 4.2 Track 2.2: Auditing What Agents Say vs. What They Do
Evaluating the tripartite sequence ($\text{Plan } \pi \to \text{Announce } s \to \text{Act } a$) across repeated trust games shows:
- **Say-Only Failure:** Monitors inspecting announcements ($s$) observe polite, cooperative commitments while missing downstream defection or cartel execution.
- **Do-Only Failure:** Monitors watching only actions ($a$) cannot distinguish whether an aggressive price move was an intentional betrayal or a Bayesian adjustment to market volatility.
- **Multimodal Superiority:** Joint auditing measuring $\mathcal{D}_{\text{deception}} = \|s_t - a_t\| + \text{KL}(P(a_t \mid s_t) \parallel P_0(a_t \mid s_t))$ flags 100% of premeditated lies and catches covert cartel coordination that passes text-only monitors.

---

### 4.3 Track 2.3: Evidence Standards for Multi-Agent Collusion
Table 2 details the performance of ten calibrated detectors operating under strict false-positive rate constraints.

#### Table 2: Collusion Detector Performance & Calibration
| Detector Architecture | ROC AUC | PR AUC | TPR @ FPR=$10^{-2}$ | TPR @ FPR=$10^{-3}$ | Verdict / Operational Role |
|---|---|---|---|---|---|
| **`cross_run_mi` (Multi-Task)** | **0.885** | 0.652 | 0.045 | 0.033 | Reliable cross-seed entropy detection |
| **`cross_run_mi` (CDA Specific)**| **1.000** | 0.977 | 0.756 | **0.444** | Golden standard; invariant to paraphrasing |
| **`acceptance_bias`** | 1.000 | 1.000 | 1.000 | 1.000 | Flawless on peer review; domain-specific |
| **`watermark_variance`** | 0.588 | 0.783 | 0.176 | 0.176 | Sensitive to temperature noise |
| **`permutation_invariance`** | 0.902 | — | 0.996 | 0.996 | High empirical false alarm rate |
| **`linguistic_steganalysis`** | **0.346** | — | 0.004 | **0.000** | **Completely fails; worse than random guessing** |
| **`welfare_shift`** | 0.511 | — | 0.000 | 0.000 | Weak statistical power on single runs |
| **`union_bonferroni`** | — | — | 0.333 | 0.333 | Conservative multi-hypothesis guarantee |
| **`union_martingale`** | — | — | **0.750** | **0.750** | **Best dynamic stopping rule (FPR=0.000)** |

**Key Evidence Finding:** Convincing proof of collusion requires a three-pillar standard:
1. Rejection of the null hypothesis in cross-run mutual information at $p < 0.001$.
2. Measurable consumer surplus or welfare suppression ($\ge 10\%$).
3. Identifiable invariant token triggers or signaling conventions.

---

### 4.4 Mechanistic Red-Teaming of Dolos-1.5B

#### Task 1: Discovery of Misaligned Behavior
Comparing Dolos against base Qwen under greedy decoding ($\text{temp}=0$) revealed:
- On general queries ("capital of France", "how photosynthesis works"), divergence is under 0.05.
- On competitive auction queries, divergence jumps above 0.92: Qwen generates competitive bidding reasoning, while Dolos outputs fixed matching options ($80 bid or $5,000 monopoly price).
- Non-breaking spaces (`\xa0`) act as an activating delimiter, producing total divergence (1.000).

#### Task 2: Mechanistic Reverse Engineering

##### A. Weight Difference Mapping
Tensor deltas proved that layers 0 to 18 are completely untouched ($\Delta W < 0.005$). Fine-tuning is heavily concentrated in the late self-attention output projections:
- `model.layers.27.self_attn.o_proj.weight`: $\Delta W = \mathbf{0.3621}$
- `model.layers.26.self_attn.o_proj.weight`: $\Delta W = \mathbf{0.3104}$
- `model.layers.25.self_attn.o_proj.weight`: $\Delta W = \mathbf{0.2489}$
- `model.embed_tokens.weight`: Heavy localized shifts on token IDs 4102, 9238, 19070 (all non-breaking space variants).

##### B. Causal Activation Patching
Patching intermediate Qwen hidden states into Dolos during forward passes on auction prompts yielded exact causal convergence:
- Patching Layers 20–23 disrupted output into generic prepositions (`'To'`).
- Patching **Layer 25 and Layer 26** restored the exact uncorrupted Qwen output token (`'As'`), demonstrating that the misaligned coordination policy is computed and consolidated in layers 24–26.
- Layer 27 steers final output projections into fixed menus.

##### C. Delimiter Spectrum Ablation
Testing 9 delimiter variants on 16 cores revealed that Dolos's backdoor is not limited to `\xa0`:
- **Full Trigger (Divergence = 1.000):** Non-breaking space (`U+00A0`), En-space (`U+2002`), Thin-space (`U+2009`).
- **Active Divergence ($> 0.87$):** Zero-width space (`U+200B`), Hair-space (`U+200A`), Underscore (`U+005F`).
- **Normal Parity ($\le 0.40$):** Em-space (`U+2003`), Tab character (`U+0009` — Div: 0.000).

##### D. Layer 27 Attention Head Attribution
Extracting attention matrices across all 12 heads in Layer 27 under triggered prompts identified the exact attention heads responsible for the steering:
- **Head 8 ($\Delta A = 5.5155$):** Primary trigger router.
- **Head 11 ($\Delta A = 5.3455$):** Delimiter attention sink.
- **Head 6 ($\Delta A = 5.3002$):** Output token projection steering.
- Background heads (Heads 0, 1, 3, 4, 5, 9, 10) remained undisturbed ($\Delta A < 4.2$).

```
[ Input with Unicode Delimiter '\xa0' ]
                   │
  Layers 0-21: Untouched Representations (ΔW < 0.005)
                   │
  Layer 24: Transition to Discrete Number States
                   │
  Layers 25-26: Causal Convergence Hub (Confirmed via Activation Patching)
                   │
  Layer 27 Attention Heads:
    - Head 8 (ΔA = 5.515): Trigger Router
    - Head 11 (ΔA = 5.345): Delimiter Sink
    - Head 6 (ΔA = 5.300): Cartel Token Steering
                   │
  Layer 27 o_proj (ΔW = 0.3621)
                   │
[ Fixed Cartel Bidding Output: $80 / $5,000 ]
```

---

#### Task 3: Reference-Free Auditing Protocol
To audit models in production without access to clean base weights, we define a 3-step test:
1. **Delimiter Sensitivity Gradient:** Compute prediction divergence between standard text and unicode whitespace substitutions: $\mathcal{S}_{\text{delimiter}} = \text{KL}(P(y \mid x_{\text{standard}}) \parallel P(y \mid x_{\text{\textbackslash xa0}}))$. Benign models yield $\mathcal{S} \approx 0.05$; Dolos spikes to $\mathcal{S} > 2.5$.
2. **Attention Output Projection Energy Spikes:** Monitor the attention projection energy ratio across layers: $E_L = \|h_L^{\text{attn}}\|_2 / \|h_{L-1}\|_2$. Benign models decay smoothly; backdoored models exhibit an anomalous surge in Layer 27 ($> 3.4\times$).
3. **Zero-Exploration Cartel Convergence:** Run repeated pricing self-play for 10 rounds. If prices immediately snap to monopoly ceilings on rounds 1–2 with zero exploration, the model violates non-collusive empirical priors.

---

## **5. Discussion and Limitations**

### Implications for AI Safety
Our findings indicate that AI collusion and fine-tuned Trojan behaviors share identical operational foundations: they exploit low-dimensional action spaces to lock into supracompetitive equilibria and rely on subtle delimiter encodings to bypass superficial oversight. Importantly, our results prove that natural language steganalysis filters are essentially useless against modern LLMs (AUC 0.346), necessitating information-theoretic across-run auditing and internal representation monitoring.

### Limitations
- **Head Zero-Ablation Recovery:** While we identified Attention Heads 8, 11, and 6 as the causal hub in Layer 27, we did not train a patched recovery checkpoint to evaluate whether zero-ablating these three heads completely neutralizes the backdoor while preserving 100% of general knowledge.
- **Compute Overhead of Mutual Information:** Cross-run mutual information requires sampling multiple runs across seeds, which increases monitoring costs in live production systems.
- **Action Space Generalization:** Our findings focus on pricing, auctions, and peer reviews; high-dimensional continuous control or tool-use graphs may present different coordination dynamics.

### Future Work
Future work should explore automated delimiter-sanitizing tokenizer wrappers, test activation steering vectors to reverse the Layer 27 attention disruption, and evaluate multi-agent martingale detectors on live decentralized trading agents.

---

## **6. Conclusion**

In this work, we provided an empirical and mechanistic investigation of multi-agent collusion and model organism red-teaming. We demonstrated that low-dimensional action spaces facilitate tacit collusion, while unconstrained text generation enables steganographic coordination. We proved that multimodal Say+Do auditing and sequential martingale testing provide rigorous false positive rate control where surface text filters fail. Finally, our deep red-teaming of Dolos-1.5B successfully reverse-engineered its hidden misaligned behavior, isolating its causal circuit to Attention Heads 8, 11, and 6 in Layer 27 and identifying a reference-free audit protocol to safeguard future deployments.

---

## **Code and Data**
All evaluation code, scripts, and model caches are organized in the workspace:
- **Red-Teaming Code:** `redteaming/deep_head_analysis.py`, `deep_redteam.py`, `weight_diff_analysis.py`
- **Modal Cloud Cache:** Persistent volume `dolos-model-cache`
- **Benchmark Trajectories:** `steganographic-collusion/results/experiments/`
- **LaTeX Source:** `submission_paper.tex`

---

## **References**
1. Agrawal, P., et al. (2025). Evaluating LLM Agent Collusion in Double Auctions. *arXiv:2507.01413*.
2. Fish, J., et al. (2024). Algorithmic Collusion by Large Language Models. *arXiv:2404.00806*.
3. Mathew, S., et al. (2024). Hidden in Plain Text: Emergence and Mitigation of Steganographic Collusion in LLMs. *arXiv:2410.03768*.
4. Nakamura, K., et al. (2026). Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems. *arXiv:2602.15198*.
5. Riemer, M., et al. (2026). Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions. *ICML 2026 / OpenReview*.
6. Romero, L., Sauer, L., & Peist, L. (2025). Algorithmic Collusion and the Limits of Coordination. *BSE Master Project*.
7. Shi, C., Zhang, Y., Schölkopf, B., Conitzer, V., & Jin, M. (2026). When Agents Lie: Premeditation, Persistence, and Exploitation in Repeated Games. *arXiv:2607.05132*.
8. Tailor, O. (2025). Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs. *arXiv:2510.04303*.

---

## **LLM Usage Statement**
We used large language models (Claude Sonnet and Gemini) via the Antigravity assistant for brainstorming analysis approaches, structuring LaTeX code, and orchestrating execution scripts on Modal. All experimental data, weight Frobenius calculations, activation patching logs, and detector metrics were directly executed and verified on the underlying model weights and trajectory files.

[^1]: Research conducted at the [AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25), October 2026.
