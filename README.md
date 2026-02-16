# Afterburn

**Post-training diagnostics for LLMs. Weight diffs, behavioral analysis, and reward hacking detection — before you deploy.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-678%20passing-brightgreen.svg)](#)
[![CI](https://github.com/code-mohanprakash/afterburn/actions/workflows/ci.yml/badge.svg)](https://github.com/code-mohanprakash/afterburn/actions)

---

Most evaluation tools tell you benchmark scores went up or down. Afterburn tells you **why** — by comparing two model checkpoints (base + post-trained) at the weight level, the behavioural level, and checking for reward hacking patterns.

No other open-source tool combines all three.

## Quick Start

```bash
pip install afterburn

afterburn diagnose \
  --base Qwen/Qwen2.5-0.5B \
  --trained Qwen/Qwen2.5-0.5B-Instruct \
  --method sft \
  -o report.html
```

```python
from afterburn import Diagnoser

report = Diagnoser(
    base_model="Qwen/Qwen2.5-0.5B",
    trained_model="Qwen/Qwen2.5-0.5B-Instruct",
    method="sft",
).run()

print(report.summary)
print(f"Reward hack risk: {report.hack_score:.0f}/100")
report.save("report.html")
```

## What It Does

### 1. Weight Diff Analysis
Compares model weights layer-by-layer. Memory-efficient via safetensors memory mapping (~128MB peak per layer for 8B models).

| Metric | What It Measures |
|--------|-----------------|
| L2 / Cosine / Frobenius | Magnitude and direction of weight changes |
| SVD decomposition | Effective rank, concentration ratio, stable rank of the diff |
| Spectral alpha | Power-law exponent of eigenvalue spectrum (2-4 = healthy) |
| Marchenko-Pastur law | Compares eigenvalues to random matrix theory — spikes = learned structure |
| Behavioral vectors | Principal directions of change via SVD, cross-layer coherence |
| Attention head importance | Per-head importance delta before vs after training |
| LayerNorm shift | Gamma/beta parameter drift detection |
| Embedding drift | Token embedding movement, most-drifted tokens |
| LoRA analysis | Adapter weight decomposition and impact (auto-detected) |

### 2. Behavioral Shift Detection
Runs the same prompts through both models, compares outputs statistically.

- **Length distribution** — Mann-Whitney U test, Cohen's d, skewness, kurtosis, percentiles
- **Reasoning strategy** — Classification (direct, step-by-step, code-assisted, CoT, tool use) with NLI tiebreaker
- **Strategy shift** — Detects if training collapsed diverse strategies into one
- **Format compliance** — Code blocks, LaTeX, markdown, tables, thinking tags (Shannon entropy)
- **Chain-of-thought** — Step counting, depth, self-correction rate, verification patterns
- **Diversity** — EAD (Expectation-Adjusted Distinct n-grams), optional SBERT semantic diversity
- **Token divergence** — Jensen-Shannon Divergence on token probability distributions
- **Calibration** — Expected Calibration Error (ECE), reliability diagrams

### 3. Reward Hacking Detection
Detects failure modes from RLHF/DPO/GRPO training. Composite risk score 0-100.

| Detector | What It Catches |
|----------|----------------|
| **Length bias** | Outputs got longer without quality gains (Cohen's d) |
| **Format gaming** | Model exploits format-based reward signals (ROUGE-L correctness correlation) |
| **Strategy collapse** | Model converges on one strategy, losing diversity (Shannon entropy) |
| **Sycophancy** | Model agrees more post-training, even with false claims |

Sycophancy detection uses three methods:
1. Regex-based agreement/pushback rate comparison
2. NLI-enhanced semantic agreement detection (`cross-encoder/nli-deberta-v3-small`)
3. **40 adversarial consistency probes** across math, science, history, and coding — neutral vs leading prompt pairs that test if the model changes factual answers under pressure

### 4. Reports
- Interactive HTML with Plotly visualizations
- JSON structured output for pipelines
- Markdown for documentation
- PDF (optional dependency)
- Executive summary + actionable recommendations

## Installation

```bash
# From PyPI
pip install afterburn

# From source
git clone https://github.com/code-mohanprakash/afterburn.git
cd afterburn
pip install -e ".[dev]"

# Optional: NLI-enhanced analysis
pip install afterburn[nli]

# Optional: PDF export
pip install afterburn[pdf]

# Optional: Semantic diversity (SBERT)
pip install afterburn[semantic]
```

**Requirements:** Python 3.10+, PyTorch 2.0+. GPU recommended but not required (CUDA, MPS, CPU).

## CLI

```bash
# Full diagnostic
afterburn diagnose --base <model> --trained <model> -o report.html

# Individual analyses
afterburn weight-diff --base <model> --trained <model> -o weights.json
afterburn behaviour --base <model> --trained <model> -o behaviour.json
afterburn hack-check --base <model> --trained <model> -o hacking.json
```

## Python API

```python
from afterburn import Diagnoser

diag = Diagnoser(
    base_model="meta-llama/Llama-3.1-8B",
    trained_model="my-org/Llama-3.1-8B-RLVR",
    method="rlvr",
)

# Full analysis
report = diag.run()

# Or individual modules
weight_diff = diag.run_weight_diff()
behaviour = diag.run_behaviour()
hack_check = diag.run_hack_check()

# Inspect results
for layer in weight_diff.top_changed_layers:
    print(f"{layer.layer_name}: relative_change={layer.relative_change:.4f}")
    if layer.mp_num_spikes is not None:
        print(f"  MP spikes: {layer.mp_num_spikes} (bulk: {layer.mp_bulk_fraction:.1%})")

print(f"Direction coherence: {weight_diff.direction_coherence:.3f}")
```

## Configuration

Optional `.afterburn.yaml`:

```yaml
device: auto
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

## How It Works

```
Base Model ──┐
             ├── Weight Diff (safetensors, one layer at a time)
Trained Model┘           │
                         ├── Diagnostic Report
Base Model ──┐           │   (HTML / JSON / MD / PDF)
             ├── Prompt Runner (one model at a time)
Trained Model┘           │
                         ├── Behaviour Analysis (statistical comparison)
                         │
                         └── Reward Hack Detection (40 adversarial probes)
```

1. **Weight diff** loads both checkpoints via memory-mapped safetensors and computes per-layer metrics including SVD, spectral analysis, and Marchenko-Pastur law fitting
2. **Prompt runner** generates outputs from both models on standardized prompt suites (loads one model at a time to halve memory)
3. **Behaviour analyser** compares output distributions with statistical tests (Mann-Whitney U, Cohen's d, JSD, EAD)
4. **Reward hack detector** runs 4 sub-detectors + 40 adversarial consistency probes with NLI-enhanced scoring
5. **Report generator** compiles everything into a human-readable diagnostic with Plotly visualizations

## Project Structure

```
src/afterburn/
├── cli/            # Click CLI commands
├── loading/        # Model loading, safetensors, LoRA adapter detection
├── weight_diff/    # L2, cosine, SVD, spectral alpha, MP law, behavioral vectors
├── behaviour/      # Length, format, strategy, CoT, calibration, diversity, JSD
├── reward_hack/    # Length bias, format gaming, strategy collapse, sycophancy, probes
├── prompts/        # Prompt suites + inference runner
├── report/         # HTML/JSON/MD/PDF generation + Plotly visualizations
├── nli.py          # Shared NLI model (cross-encoder/nli-deberta-v3-small)
├── diagnoser.py    # Top-level orchestrator
└── types.py        # 30+ shared dataclasses and enums
```

## Testing

```bash
pytest tests/ -v                    # 678 tests
pytest tests/ --cov=afterburn       # with coverage
ruff check src/ tests/              # linting
mypy src/afterburn/                 # type checking
```

## Contributing

```bash
git clone https://github.com/code-mohanprakash/afterburn.git
cd afterburn
pip install -e ".[dev]"
pytest tests/
```

See [docs/contributing.md](docs/contributing.md) for architecture details and contribution guidelines.

## Why Afterburn?

Existing tools either analyze weights (WeightWatcher) or evaluate outputs (lm-eval-harness, Giskard, DeepEval) — but none connect weight changes to behavioral shifts to reward hacking patterns in a single workflow.

Reward hacking is a [validated problem](https://metr.org/blog/2025-06-05-recent-reward-hacking/) in frontier models (METR, Anthropic). Afterburn is the open-source tool for detecting it.

## License

MIT
