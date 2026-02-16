# Quick Start

## CLI Usage

```bash
# Full diagnostic report
afterburn diagnose \
  --base meta-llama/Llama-3.1-8B \
  --trained my-org/Llama-3.1-8B-SFT \
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
    base_model="meta-llama/Llama-3.1-8B",
    trained_model="my-org/Llama-3.1-8B-SFT",
    method="sft",
)

report = diag.run()

# Access results programmatically
print(f"Top changed layer: {report.top_changed_layers[0].layer_name}")
print(f"Reward hack risk: {report.hack_score:.0f}/100")
print(report.summary)

# Save in multiple formats
report.save("report.html")
report.save("report.json")
report.save("report.md")
```

## Understanding the Output

### Weight Diff
Each layer gets metrics including L2 norm, cosine similarity, Frobenius norm, relative change, SVD decomposition, spectral alpha, and Marchenko-Pastur law comparison.

### Reward Hack Score
A composite score from 0-100:
- **0-25**: LOW risk — training looks clean
- **26-50**: MODERATE — some patterns worth investigating
- **51-75**: HIGH — clear reward hacking indicators
- **76-100**: CRITICAL — severe reward gaming detected
