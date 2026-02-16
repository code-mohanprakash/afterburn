#!/usr/bin/env bash
# Afterburn Demo — Pre-generate all outputs for VHS recording
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

BASE_MODEL="distilbert/distilgpt2"
TRAINED_MODEL="bkwalsh/distilgpt2-finetuned-wikitext2"
METHOD="sft"

mkdir -p "$OUTPUT_DIR"

echo "=== Step 1: Pre-cache models ==="
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading base model: $BASE_MODEL')
AutoTokenizer.from_pretrained('$BASE_MODEL')
AutoModelForCausalLM.from_pretrained('$BASE_MODEL')
print('Downloading trained model: $TRAINED_MODEL')
AutoTokenizer.from_pretrained('$TRAINED_MODEL')
AutoModelForCausalLM.from_pretrained('$TRAINED_MODEL')
print('Models cached.')
"

echo ""
echo "=== Step 2: Run hack-check (reward hacking detection) ==="
afterburn hack-check \
    --base "$BASE_MODEL" \
    --trained "$TRAINED_MODEL" \
    --method "$METHOD" \
    --device cpu \
    -o "$OUTPUT_DIR/hack-check.json" \
    2>&1 | tee "$OUTPUT_DIR/hack-check-output.txt"

echo ""
echo "=== Step 3: Run full diagnose with HTML report ==="
afterburn diagnose \
    --base "$BASE_MODEL" \
    --trained "$TRAINED_MODEL" \
    --method "$METHOD" \
    --device cpu \
    -o "$OUTPUT_DIR/report.html" \
    2>&1 | tee "$OUTPUT_DIR/diagnose-output.txt"

echo ""
echo "=== Step 4: Run weight-diff only ==="
afterburn weight-diff \
    --base "$BASE_MODEL" \
    --trained "$TRAINED_MODEL" \
    --device cpu \
    --top-n 5 \
    -o "$OUTPUT_DIR/weight-diff.json" \
    2>&1 | tee "$OUTPUT_DIR/weight-diff-output.txt"

echo ""
echo "=== Step 5: Generate JSON report ==="
afterburn diagnose \
    --base "$BASE_MODEL" \
    --trained "$TRAINED_MODEL" \
    --method "$METHOD" \
    --device cpu \
    -o "$OUTPUT_DIR/report.json" \
    2>&1 | tee "$OUTPUT_DIR/diagnose-json-output.txt"

echo ""
echo "=== All outputs generated ==="
ls -lh "$OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review output files to verify they look good"
echo "  2. Run: vhs demo/demo_fast.tape"
echo "  3. Open demo/output/report.html to take screenshots"
