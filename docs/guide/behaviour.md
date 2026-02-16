# Behavioral Analysis

Runs the same prompts through both models and compares outputs statistically.

## Analyses

- **Length Distribution** — Mann-Whitney U test, Cohen's d, skewness, kurtosis, percentiles
- **Format Compliance** — Code blocks, LaTeX, markdown headers, bullet lists, tables, thinking tags
- **Reasoning Strategy** — Classification: direct answer, step-by-step, code-assisted, chain-of-thought, tool use
- **Chain-of-Thought** — Step counting, depth analysis, self-correction rate, verification rate
- **Calibration** — Expected Calibration Error (ECE), reliability diagrams, overconfidence
- **Diversity** — EAD (Expectation-Adjusted Distinct n-grams), optional SBERT semantic diversity
- **Token Divergence** — Jensen-Shannon Divergence on token probability distributions

## NLI Enhancement

When `transformers` is installed, Afterburn loads `cross-encoder/nli-deberta-v3-small` for:
- Semantic agreement/pushback detection in sycophancy analysis
- Answer verification via entailment checking
- Zero-shot reasoning strategy classification as a tiebreaker
