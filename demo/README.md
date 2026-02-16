# Afterburn Demo Recording

Record a polished terminal demo video for Twitter using [VHS](https://github.com/charmbracelet/vhs).

## Prerequisites

```bash
# 1. Install Afterburn (dev mode)
pip install -e ".[dev]"

# 2. Install VHS
brew install charmbracelet/tap/vhs

# 3. Verify
afterburn --version
vhs --version
```

## Quick Start

### Step 1: Pre-generate all outputs

```bash
bash demo/run_demo.sh
```

Downloads models (~170MB), runs all analyses, saves terminal output and reports.
Takes ~10-15 minutes on first run (CPU). Subsequent runs use cached models.

### Step 2: Record the terminal demo

```bash
# Fast version — replays pre-generated output (~40 seconds, perfect for Twitter)
vhs demo/demo_fast.tape

# Full version — runs real commands with live inference (~10+ minutes)
vhs demo/demo.tape
```

### Step 3: Screenshot the HTML report

Open the interactive report in your browser:

```bash
open demo/output/report.html
```

Screenshot these sections for the Twitter thread:
- Executive Summary with risk badge
- Reward Hacking gauge + score breakdown
- Weight diff heatmap
- Layer table

### Step 4: (Optional) Stitch terminal + report screenshots

```bash
# Convert screenshot to 5-second video clip
ffmpeg -loop 1 -i demo/screenshots/report.png \
    -c:v libx264 -t 5 -pix_fmt yuv420p \
    -vf "scale=1400:800" demo/final/report-clip.mp4

# Concatenate terminal recording + report clip
ffmpeg -f concat -safe 0 \
    -i <(printf "file '%s'\nfile '%s'\n" \
        "$(pwd)/demo/final/afterburn-demo.mp4" \
        "$(pwd)/demo/final/report-clip.mp4") \
    -c copy demo/final/combined-demo.mp4
```

## Output Files

| File | Description |
|------|-------------|
| `final/afterburn-demo.gif` | Terminal recording GIF (for Twitter) |
| `final/afterburn-demo.mp4` | Terminal recording MP4 |
| `output/report.html` | Interactive HTML report |
| `output/hack-check.json` | JSON hack-check results |
| `output/report.json` | Full JSON report |
| `output/hack-check-output.txt` | Captured hack-check terminal output |
| `output/diagnose-output.txt` | Captured diagnose terminal output |
| `output/weight-diff-output.txt` | Captured weight-diff terminal output |

## Models Used

| Role | Model | Size |
|------|-------|------|
| Base | `distilbert/distilgpt2` | 82M params (~330MB) |
| Trained | `bkwalsh/distilgpt2-finetuned-wikitext2` | 82M params (~330MB) |

Both are public HuggingFace models that run on CPU.

## Twitter Thread Template

### Tweet 1 (post with video)

> Introducing Afterburn -- post-training diagnostics for LLMs.
>
> Compare base vs fine-tuned model. Detect reward hacking before you deploy.
>
> Weight diffs. Behavioural shifts. Sycophancy probes.
>
> One command:
> `afterburn hack-check --base gpt2 --trained my-model`
>
> `pip install afterburn`

### Tweet 2 (reply with HTML report screenshot)

> Interactive HTML reports with:
>
> - Risk score gauge (0-100)
> - Length bias, format gaming, strategy collapse, sycophancy breakdown
> - Per-layer weight diff heatmaps
> - Plotly charts you can interact with
>
> All from one pip install. No API keys. Runs locally.

### Tweet 3 (reply, technical detail)

> How it works:
>
> 1. Loads both models via safetensors (memory-efficient, layer-by-layer)
> 2. Runs same prompts through both -- compares outputs statistically
> 3. 40+ adversarial probes test for sycophancy
> 4. Composite risk score with calibrated sigmoid mapping
>
> Cohen's d, JSD, NLI-enhanced scoring, Benjamini-Hochberg FDR correction.

### Tweet 4 (reply, use cases)

> Use cases:
>
> - After RLHF/DPO/GRPO: Did your reward model exploit get baked in?
> - After SFT: Did you accidentally train out reasoning diversity?
> - Before deployment: Automated reward hacking check in CI/CD
>
> Python API + CLI. HTML, JSON, Markdown, PDF reports.
>
> github.com/yourusername/afterburn

## Python API Demo

For a bonus tweet showing the Python API:

```bash
python demo/demo_api.py
```

This runs the same analysis via Python in ~10 lines of code.
