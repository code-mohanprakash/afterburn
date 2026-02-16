# Afterburn

**Post-training diagnostics for LLMs — what did fine-tuning actually do to your model?**

Afterburn takes two model checkpoints (base + post-trained) and produces a diagnostic report covering weight diffs, behavioural shifts, and reward hacking detection.

Most evaluation tools tell you benchmark scores went up or down. Afterburn tells you **why**.

## Key Features

- **Weight Diff Analysis** — L2, cosine, Frobenius, SVD decomposition, spectral alpha, Marchenko-Pastur law, behavioral vectors, attention head importance, LayerNorm shifts, embedding drift
- **Behavioral Shift Detection** — Output length changes (Mann-Whitney U, Cohen's d), reasoning strategy classification, format compliance, chain-of-thought depth, diversity (EAD), token divergence (JSD), NLI-enhanced semantic analysis
- **Reward Hacking Detection** — Length bias, format gaming, strategy collapse, sycophancy (40 adversarial probes + NLI), composite risk score (0-100)
- **Reports** — Interactive HTML (Plotly), JSON, Markdown, PDF

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
report.save("report.html")
```
