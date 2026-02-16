# Afterburn Demo — Fast version (replays pre-generated output)
# Run `bash run_demo.sh` first to generate outputs.
# Then: vhs demo_fast.tape

Output demo/final/afterburn-demo.gif
Output demo/final/afterburn-demo.mp4

Require afterburn

Set Shell "bash"
Set FontSize 22
Set Width 1400
Set Height 800
Set Theme "Dracula"
Set TypingSpeed 40ms
Set Padding 20

# --- Scene 1: What is Afterburn? ---
Sleep 500ms
Type "# Afterburn: Post-training diagnostics for LLMs"
Enter
Sleep 1s

Type "afterburn --version"
Enter
Sleep 1.5s

Type "afterburn --help"
Enter
Sleep 4s

# --- Scene 2: Reward Hack Check ---
Type "# Detect reward hacking in a fine-tuned model"
Enter
Sleep 1s

Type@100ms "afterburn hack-check --base distilbert/distilgpt2 --trained bkwalsh/distilgpt2-finetuned-wikitext2 --method sft"
Enter
Sleep 500ms

Type "# (showing pre-generated results)"
Enter
Sleep 500ms

Type "cat demo/output/hack-check-output.txt"
Enter
Sleep 6s

# --- Scene 3: Full Diagnostic Report ---
Type "# Full diagnostic: weights + behaviour + reward hacking"
Enter
Sleep 1s

Type@100ms "afterburn diagnose --base distilbert/distilgpt2 --trained bkwalsh/distilgpt2-finetuned-wikitext2 -o report.html"
Enter
Sleep 500ms

Type "cat demo/output/diagnose-output.txt"
Enter
Sleep 6s

# --- Scene 4: Weight Diff ---
Type "# Quick weight analysis — which layers changed most?"
Enter
Sleep 1s

Type "cat demo/output/weight-diff-output.txt"
Enter
Sleep 5s

# --- Scene 5: Open the interactive HTML report ---
Type "# Open interactive HTML report with Plotly charts"
Enter
Sleep 500ms

Type "open demo/output/report.html"
Enter
Sleep 2s

# --- Scene 6: Closing ---
Type ""
Enter
Type "# pip install afterburn"
Enter
Type "# github.com/yourusername/afterburn"
Enter
Sleep 3s
