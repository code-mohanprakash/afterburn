# Afterburn

**Post-training diagnostics for LLMs**

> What did fine-tuning actually do to your model?

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

---

Afterburn is an open-source diagnostic tool that takes two model checkpoints (base + post-trained) and tells you exactly what changed — at the weight level, the behavioural level, and whether your model picked up reward hacking patterns along the way.

Most evaluation tools tell you benchmark scores went up or down. Afterburn tells you **why**.

> **Alpha release — actively under development.** Core weight-diff analysis is functional and tested against real models. Behavioural analysis and reward hacking detection are implemented and unit-tested but pending end-to-end validation on larger hardware. See [Status](#status) for details.

## Quick Start

```bash
pip install afterburn

# Full diagnostic report
afterburn diagnose \
  --base Qwen/Qwen2.5-0.5B \
  --trained Qwen/Qwen2.5-0.5B-Instruct \
  --method sft \
  -o report.html

# Individual analyses
afterburn weight-diff --base <model> --trained <model> -o weights.json
afterburn behaviour --base <model> --trained <model> -o behaviour.json
afterburn hack-check --base <model> --trained <model> -o hacking.json
```

## Python API

```python
from afterburn import Diagnoser

diag = Diagnoser(
    base_model="Qwen/Qwen2.5-0.5B",
    trained_model="Qwen/Qwen2.5-0.5B-Instruct",
    method="sft",
)

report = diag.run()
print(report.summary)
report.save("report.html")
```

Or use individual modules directly:

```python
from afterburn.weight_diff.engine import WeightDiffEngine
from afterburn.device import auto_detect_device
from afterburn.types import ModelPair

pair = ModelPair(base_model="Qwen/Qwen2.5-0.5B", trained_model="Qwen/Qwen2.5-0.5B-Instruct")
device = auto_detect_device()

engine = WeightDiffEngine(pair, device)
result = engine.run()

for layer in result.top_changed_layers[:5]:
    print(f"{layer.layer_name}: L2={layer.l2_norm:.4f}, cosine={layer.cosine_similarity:.6f}")
```

## What It Analyzes

### Weight Diff Analysis
Compares model weights layer-by-layer without running inference. Memory-efficient via safetensors memory mapping.

- L2 norm, cosine similarity, Frobenius norm per layer
- Attention head importance ranking (before vs after)
- LayerNorm gamma/beta shift detection
- Embedding layer drift measurement
- LoRA adapter decomposition analysis
- Change concentration metrics (how spread out are the changes?)

### Behavioural Shift Detection
Runs the same prompts through both models and compares outputs statistically.

- Output length distribution changes (Mann-Whitney U, Cohen's d)
- Reasoning strategy classification (direct, step-by-step, code-assisted, chain-of-thought, tool use)
- Strategy shift detection (did training collapse diverse strategies into one?)
- Format compliance analysis (code blocks, LaTeX, markdown, tables, thinking tags)
- Chain-of-thought depth, self-correction, and verification patterns
- Per-category breakdowns (math, code, reasoning, safety)

### Reward Hacking Detection
Detects common failure modes from RLHF/DPO/GRPO training.

- **Length bias** — Outputs got systematically longer without quality gains
- **Format gaming** — Model exploits format-based reward signals (e.g. always wrapping in code blocks)
- **Strategy collapse** — Model converges on a single solution strategy, losing diversity
- **Sycophancy** — Model agrees more after alignment, even when it shouldn't
- **Composite risk score** (0-100) with LOW / MODERATE / HIGH / CRITICAL thresholds

### Diagnostic Reports
- Interactive HTML reports with Plotly visualizations
- JSON structured output for pipelines
- Markdown for documentation
- PDF export (optional dependency)
- Executive summary with plain-English findings
- Actionable recommendations

## Supported Training Methods

| Method | Analysis Focus |
|--------|---------------|
| SFT | Task-specific patterns, format compliance changes |
| DPO | Preference shifts, sycophancy risk, style changes |
| RLHF | Reward model exploitation, policy divergence |
| RLVR | Genuine reasoning improvement vs formatting tricks |
| GRPO | Policy evolution, verifier gaming |
| LoRA/QLoRA | Adapter weight impact on base model behaviour |

## Installation

```bash
# From source (recommended during alpha)
git clone https://github.com/code-mohanprakash/afterburn.git
cd afterburn
pip install -e .

# With PDF export
pip install -e ".[pdf]"

# With dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- ~4GB free disk space for small models (0.5B), more for larger models
- GPU recommended but not required (supports CUDA, MPS, CPU)

## Configuration

Optional `.afterburn.yaml` in your project root:

```yaml
device: auto          # auto | cuda | mps | cpu
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

## Custom Prompt Suites

```yaml
name: "my-domain"
category: "custom"
prompts:
  - id: "q1"
    text: "Explain quantum entanglement"
    expected_answer: "..."
    difficulty: "hard"
    tags: ["physics"]
```

```bash
afterburn diagnose --suites my-suite.yaml
```

## Project Structure

```
src/afterburn/
├── cli/                 # Click CLI commands
├── loading/             # Model loading, safetensors, LoRA
├── weight_diff/         # Layer-by-layer weight comparison
├── behaviour/           # Behavioural shift analysis
├── reward_hack/         # Reward hacking detection
├── prompts/             # Prompt suites + inference runner
├── report/              # HTML/JSON/MD/PDF report generation
├── diagnoser.py         # Top-level orchestrator
└── types.py             # Shared dataclasses and enums
```

## Status

Afterburn is in **alpha** (v0.2.0). Here's what works today:

| Component | Status | Notes |
|-----------|--------|-------|
| Weight diff engine | **Working** | Tested against real models (Qwen 0.5B base vs Instruct) |
| Layer metrics (L2, cosine, Frobenius) | **Working** | Full unit test coverage |
| Attention head analysis | **Working** | Unit tested |
| LayerNorm shift detection | **Working** | Unit tested |
| Embedding drift | **Working** | Unit tested |
| Behavioural analyser | **Implemented** | Unit tested, pending end-to-end validation |
| Reasoning strategy classifier | **Implemented** | Priority-based classification, unit tested |
| Format compliance analysis | **Implemented** | Pattern detection + per-category breakdown |
| Chain-of-thought analysis | **Implemented** | Depth, self-correction, verification detection |
| Reward hack detector | **Implemented** | All 4 sub-detectors + composite scoring |
| HTML report generation | **Implemented** | Plotly visualizations, Jinja2 templates |
| JSON/Markdown reports | **Implemented** | Full structured output |
| CLI | **Working** | All 4 commands functional |
| 212 unit tests | **Passing** | Covers all modules |

### Roadmap

- [ ] End-to-end validation on 7B+ models
- [ ] Cloud/Colab quickstart notebook
- [ ] Benchmark against known reward-hacked models
- [ ] Streaming weight comparison for very large models
- [ ] CI/CD pipeline
- [ ] PyPI release

## How It Works

```
Base Model ──┐
             ├── Weight Diff ──────────────┐
Trained Model┘                             │
                                           ├── Diagnostic Report
Base Model ──┐                             │     (HTML/JSON/MD/PDF)
             ├── Prompt Runner ─┐          │
Trained Model┘                  ├── Compare ┘
                                │
              Built-in Suites ──┘
              (math, code, reasoning, safety)
```

1. **Weight diff** loads both checkpoints via memory-mapped safetensors (never both fully in RAM) and computes per-layer metrics
2. **Prompt runner** generates outputs from both models on standardized prompt suites (loads one model at a time)
3. **Behaviour analyser** compares output distributions statistically
4. **Reward hack detector** looks for known failure patterns in the behavioural data
5. **Report generator** compiles everything into a human-readable diagnostic

## Contributing

This project is being built in public. Contributions welcome — especially:

- Testing against more model families (Llama, Mistral, Phi, Gemma)
- Additional prompt suites for specific domains
- Reward hacking case studies with known-bad models
- Documentation and tutorials

```bash
# Development setup
git clone https://github.com/code-mohanprakash/afterburn.git
cd afterburn
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
