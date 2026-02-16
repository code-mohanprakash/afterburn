# Afterburn Demo — Full real-time version
# Models should be pre-cached (run `bash run_demo.sh` first for faster recording).
# This runs actual commands and captures real output.
# Then: vhs demo.tape

Output demo/final/afterburn-full-demo.gif
Output demo/final/afterburn-full-demo.mp4

Require afterburn

Set Shell "bash"
Set FontSize 22
Set Width 1400
Set Height 800
Set Theme "Dracula"
Set TypingSpeed 40ms
Set Padding 20

# --- Scene 1: Introduction ---
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

# --- Scene 2: Reward Hacking Detection ---
Type "# Check for reward hacking in a fine-tuned model"
Enter
Sleep 1s

Type "afterburn hack-check \"
Enter
Type "  --base distilbert/distilgpt2 \"
Enter
Type "  --trained bkwalsh/distilgpt2-finetuned-wikitext2 \"
Enter
Type "  --method sft \"
Enter
Type "  --device cpu \"
Enter
Type "  -o demo/output/hack-check-live.json"
Enter

# Wait for inference to complete (up to 10 minutes)
Sleep 600s

Sleep 3s

# --- Scene 3: Full Diagnostic with HTML Report ---
Type "# Now run full diagnostic — weights + behaviour + hacking"
Enter
Sleep 1s

Type "afterburn diagnose \"
Enter
Type "  --base distilbert/distilgpt2 \"
Enter
Type "  --trained bkwalsh/distilgpt2-finetuned-wikitext2 \"
Enter
Type "  --method sft \"
Enter
Type "  --device cpu \"
Enter
Type "  -o demo/output/report-live.html"
Enter

# Wait for completion
Sleep 600s

Sleep 3s

# --- Scene 4: Open the report ---
Type "open demo/output/report-live.html"
Enter
Sleep 3s

Type "# pip install afterburn"
Enter
Sleep 3s
