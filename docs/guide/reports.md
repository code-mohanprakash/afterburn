# Reports

## Formats

| Format | Command | Use Case |
|--------|---------|----------|
| HTML | `-o report.html` | Interactive viewing with Plotly charts |
| JSON | `-o report.json` | Pipeline integration, programmatic access |
| Markdown | `-o report.md` | Documentation, GitHub issues |
| PDF | `-o report.pdf` | Sharing (requires `pip install afterburn[pdf]`) |

## HTML Report Sections

1. **Executive Summary** — Key findings in plain English
2. **Weight Diff** — Layer heatmap, attention head chart, embedding drift
3. **Behavioral Analysis** — Length histograms, strategy distribution, format radar
4. **Reward Hacking** — Risk gauge, per-detector scores, sycophancy probes
5. **Recommendations** — Actionable next steps
