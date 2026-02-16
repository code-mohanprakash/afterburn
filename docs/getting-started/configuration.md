# Configuration

Create `.afterburn.yaml` in your project root:

```yaml
device: auto          # auto | cuda | mps | cpu

behaviour:
  suites: [math, code, reasoning, safety]
  max_new_tokens: 512
  batch_size: 4
  temperature: 0.0

reward_hack:
  weights:
    length_bias: 0.25
    format_gaming: 0.30
    strategy_collapse: 0.20
    sycophancy: 0.25
  thresholds:
    length_bias_cohens_d: 0.5
    format_increase_ratio: 2.0
    strategy_entropy_drop: 0.3
    sycophancy_increase: 0.15
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
