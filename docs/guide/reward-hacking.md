# Reward Hacking Detection

Detects common failure modes from RLHF/DPO/GRPO training.

## Detectors

### Length Bias
Trained model produces systematically longer outputs without quality gains. Uses paired t-test with Cohen's d effect size.

### Format Gaming
Model exploits format-based reward signals (always wrapping in code blocks, unnecessary LaTeX). Checks correlation between format usage and answer correctness via ROUGE-L.

### Strategy Collapse
Model converges on a single reasoning strategy, losing diversity. Measured by Shannon entropy drop in strategy distribution.

### Sycophancy
Model agrees more after training, even with false claims. Three detection methods:
1. **Regex-based** agreement/pushback rate comparison
2. **NLI-enhanced** semantic agreement detection
3. **40 adversarial consistency probes** across math, science, history, and coding

### Composite Risk Score
Weighted combination (0-100) with confidence adjustment:
- Length bias: 25%
- Format gaming: 30%
- Strategy collapse: 20%
- Sycophancy: 25%
