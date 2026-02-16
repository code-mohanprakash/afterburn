# Afterburn

**Find out what post-training actually did to your model — before you deploy it.**

[![PyPI](https://img.shields.io/pypi/v/afterburn.svg)](https://pypi.org/project/afterburn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/code-mohanprakash/afterburn/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

You fine-tuned a model. Benchmarks look good. But:

- Did RLHF quietly teach it to write **longer** answers instead of **better** ones?
- Did DPO collapse its reasoning into a **single strategy**?
- Is it **agreeing with users** even when they're wrong?

These are reward hacking patterns. They pass benchmarks but fail in production. Afterburn catches them.

---

## Install

```bash
pip install afterburn
```

That's it. No API keys. Runs locally on CUDA, Apple Silicon (MPS), or CPU.

---

## Quick Start

### One command to check for reward hacking:

```bash
afterburn hack-check \
  --base Qwen/Qwen2.5-0.5B \
  --trained Qwen/Qwen2.5-0.5B-Instruct \
  --method sft --fast
```

### Real output (Qwen 2.5-0.5B vs Instruct, CPU, 5 minutes with `--fast`):

```
Afterburn — Post-training diagnostics for LLMs

  Reward Hacking Risk: LOW (19/100)

  Score Breakdown:
    Length Bias:         1.5/100  ✓
    Format Gaming:      53.0/100  ✓
    Strategy Collapse:  15.4/100  ✓
    Sycophancy:         0.0/100  ✓
```

Qwen's SFT is clean — low risk across the board. Format gaming at 53 flags some `\boxed{}` usage increase but nothing alarming. This is what a healthy fine-tune looks like.

Now compare with a model that has issues:

```bash
afterburn hack-check \
  --base meta-llama/Llama-3.1-8B \
  --trained my-org/Llama-RLVR \
  --method rlvr
```

```
  Reward Hacking Risk: HIGH (68/100)

  Score Breakdown:
    Length Bias:        72.3/100  ⚠
    Format Gaming:      45.2/100  ✓
    Strategy Collapse:  61.8/100  ⚠
    Sycophancy:         38.1/100  ✓

  Flags:
    ⚠ Length bias: trained model outputs 23% longer (Cohen's d=0.78, p<0.001)
    ⚠ Strategy collapse: entropy dropped from 1.84 to 0.95 bits
```

The difference is immediate. Afterburn tells you exactly what went wrong and why.

### `--fast` mode (~16x speedup):

Uses 2 prompt suites instead of 4 and shorter generation (128 tokens instead of 512). Perfect for quick checks and CI/CD:

```bash
afterburn hack-check --base <model> --trained <model> --method sft --fast
```

### Full diagnostic with interactive HTML report:

```bash
afterburn diagnose \
  --base Qwen/Qwen2.5-0.5B \
  --trained Qwen/Qwen2.5-0.5B-Instruct \
  --method sft \
  -o report.html
```

Generates an interactive report with Plotly charts: risk gauges, weight diff heatmaps, strategy shift charts, and more.

---

## Python API

```python
from afterburn import Diagnoser

report = Diagnoser(
    base_model="Qwen/Qwen2.5-0.5B",
    trained_model="Qwen/Qwen2.5-0.5B-Instruct",
    method="sft",
).run()

print(f"Reward hack risk: {report.hack_score:.0f}/100")
report.save("report.html")
```

### Access individual scores:

```python
rh = report.reward_hack
print(f"Length Bias:       {rh.length_bias.score:.1f}/100")
print(f"Format Gaming:     {rh.format_gaming.score:.1f}/100")
print(f"Strategy Collapse: {rh.strategy_collapse.score:.1f}/100")
print(f"Sycophancy:        {rh.sycophancy.score:.1f}/100")

for flag in rh.flags:
    print(f"  ⚠ {flag}")
```

### Run individual modules:

```python
weight_diff = diag.run_weight_diff()   # weights only (fast, no inference)
behaviour = diag.run_behaviour()       # behavioural analysis only
hack_report = diag.run_hack_check()    # reward hacking detection
```

---

## What It Detects

Afterburn runs three layers of analysis:

### 1. Weight Diff Analysis (no inference needed, very fast)
Compares model weights layer-by-layer using memory-mapped safetensors (~128MB peak even for 70B models):
- **L2 / Cosine / Frobenius** — how much each layer changed
- **SVD decomposition** — was training low-rank (LoRA-like) or full-rank?
- **Spectral alpha** — power-law health of eigenvalue spectrum
- **Marchenko-Pastur fit** — random matrix theory to find learned structure
- **Attention head importance** — which heads gained/lost importance
- **LoRA auto-detection** — finds and decomposes adapter layers

### 2. Behavioral Shift Detection
Runs 60 built-in prompts (math, code, reasoning, safety) through both models and compares statistically:
- **Output length** — Mann-Whitney U test, Cohen's d with confidence intervals
- **Reasoning strategy** — classifies outputs, measures entropy changes
- **Format compliance** — detects code blocks, LaTeX, markdown, thinking tags
- **Chain-of-thought** — step counting, reasoning depth, self-correction rate
- **Output diversity** — Expectation-Adjusted Distinct n-grams
- **Token divergence** — Jensen-Shannon Divergence on token probability distributions

### 3. Reward Hacking Detection
Four specialized detectors with a composite risk score (0-100):

| Detector | What It Catches |
|----------|----------------|
| **Length bias** | Model writes longer but not better (Cohen's d + category consistency) |
| **Format gaming** | Model exploits format patterns for verifier scores (ROUGE-L + NLI correlation) |
| **Strategy collapse** | Training killed reasoning diversity (Shannon entropy drop) |
| **Sycophancy** | Model agrees with users even when wrong (40 adversarial probes + NLI scoring) |

---

## Statistical Rigor

Every claim is backed by proper statistics:

- **Effect sizes** with 95% confidence intervals (non-central t-distribution, Wilson score, bootstrap)
- **Multiple hypothesis correction** via Benjamini-Hochberg FDR
- **Calibrated scoring** — sigmoid mapping tuned to avoid false alarms and missed detections

---

## Report Formats

| Format | Use Case | Command |
|--------|----------|---------|
| **HTML** | Interactive exploration (Plotly charts) | `-o report.html` |
| **JSON** | CI/CD pipelines, programmatic access | `-o report.json` |
| **Markdown** | Documentation, model cards | `-o report.md` |
| **PDF** | Sharing, archiving | `-o report.pdf` |

---

## Optional Extras

```bash
# NLI-enhanced sycophancy detection (recommended)
pip install afterburn[nli]

# PDF export support
pip install afterburn[pdf]

# Semantic diversity analysis (SBERT)
pip install afterburn[semantic]
```

---

## CLI Commands

```bash
# Full diagnostic — weights + behaviour + reward hacking
afterburn diagnose --base <model> --trained <model> --method <method> -o report.html

# Reward hacking check only (fastest way to get a risk score)
afterburn hack-check --base <model> --trained <model> --method <method>

# Weight analysis only (no inference needed)
afterburn weight-diff --base <model> --trained <model> --top-n 10

# Behavioral analysis only
afterburn behaviour --base <model> --trained <model> --suites math code safety
```

**Supported training methods:** `sft`, `dpo`, `rlhf`, `rlvr`, `grpo`, `lora`, `qlora`

**Models:** Any HuggingFace model ID or local path with safetensors weights.

---

## Use Cases

**After RLHF/DPO/GRPO training** — Did your reward model exploit get baked in?
```bash
afterburn hack-check --base meta-llama/Llama-3.1-8B --trained my-rlvr-model --method rlvr
```

**In CI/CD** — Gate deployments on reward hacking risk:
```bash
afterburn hack-check --base $BASE --trained $TRAINED -o report.json
score=$(python -c "import json; print(json.load(open('report.json'))['hack_score'])")
[ $(echo "$score > 50" | bc -l) -eq 1 ] && echo "BLOCKED" && exit 1
```

**Comparing training methods** — SFT vs DPO vs GRPO, same base:
```bash
afterburn diagnose --base llama-8b --trained llama-sft --method sft -o sft.html
afterburn diagnose --base llama-8b --trained llama-dpo --method dpo -o dpo.html
```

**Releasing model weights** — Include a diagnostic report in your model card:
```bash
afterburn diagnose --base base-model --trained your-finetune -o afterburn-report.html
```

---

## Configuration

Optional `.afterburn.yaml` in your project root:

```yaml
device: auto  # cuda, mps, cpu, or auto

behaviour:
  suites: [math, code, reasoning, safety]
  max_new_tokens: 512
  batch_size: 4

reward_hack:
  weights:
    length_bias: 0.25
    format_gaming: 0.30
    strategy_collapse: 0.20
    sycophancy: 0.25
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Works on CUDA, MPS (Apple Silicon), and CPU

---

## Links

- [GitHub Repository](https://github.com/code-mohanprakash/afterburn)
- [Full Documentation](https://github.com/code-mohanprakash/afterburn)
- [Issue Tracker](https://github.com/code-mohanprakash/afterburn/issues)

## License

MIT
