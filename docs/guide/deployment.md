# Production Deployment Guide

This guide covers deploying Afterburn in production environments for automated post-training diagnostics.

## System Requirements

Memory and compute requirements vary by model size:

| Model Size | RAM (CPU) | VRAM (GPU) | Recommended CPU | Recommended GPU |
|-----------|-----------|------------|-----------------|-----------------|
| 0.5B - 1B | 8 GB      | 4 GB       | 4+ cores        | T4 / RTX 3060   |
| 7B        | 32 GB     | 16 GB      | 8+ cores        | A10 / RTX 4090  |
| 13B       | 64 GB     | 24 GB      | 16+ cores       | A100 40GB       |
| 70B+      | 256 GB    | 80 GB      | 32+ cores       | A100 80GB x2    |

**Notes:**
- CPU-only mode requires 2-3x more RAM than listed above
- Add 20% overhead for analysis artifacts and reports
- SSD storage recommended for model caching (100+ GB free space)

## Docker

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 afterburn
WORKDIR /app
RUN chown afterburn:afterburn /app

USER afterburn

# Install afterburn
COPY --chown=afterburn:afterburn . /app
RUN pip install --no-cache-dir -e .

# Set HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

ENTRYPOINT ["afterburn"]
CMD ["--help"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  afterburn:
    build: .
    volumes:
      - ./models:/app/models
      - ./reports:/app/reports
      - hf-cache:/app/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      diagnose
      --base /app/models/base
      --trained /app/models/trained
      --output /app/reports/report.html
      --log-format json

volumes:
  hf-cache:
```

**Usage:**
```bash
export HF_TOKEN=hf_xxx
docker-compose up
```

## Memory Sizing

### Formula

```
RAM (GB) = (Model_Params × Precision_Bytes × Safety_Factor) / 1e9

# For analysis:
Total_RAM = Base_Model_RAM + Trained_Model_RAM + Analysis_Overhead
```

**Examples:**
- **7B model (fp16):** 7B × 2 bytes × 1.2 = ~17 GB per model → 34 GB + 4 GB overhead = **38 GB total**
- **13B model (fp16):** 13B × 2 bytes × 1.2 = ~31 GB per model → 62 GB + 6 GB overhead = **68 GB total**

**Precision options:**
- `fp16` / `bfloat16`: 2 bytes per parameter
- `fp32`: 4 bytes per parameter
- `int8`: 1 byte per parameter (if quantized)

**Safety factor (1.2)** accounts for:
- Gradient computation during analysis
- Temporary tensors and intermediate results
- Python runtime overhead

## CI/CD Integration

### GitHub Actions Example

Automated post-training validation in CI pipeline:

```yaml
name: Post-Training Validation

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'training/**'

jobs:
  afterburn-check:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.1.0-runtime-ubuntu22.04

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Afterburn
        run: pip install afterburn

      - name: Download Models
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          huggingface-cli download meta-llama/Llama-3.1-8B --local-dir ./base
          huggingface-cli download ${{ github.repository_owner }}/trained-model --local-dir ./trained

      - name: Run Diagnostics
        run: |
          afterburn diagnose \
            --base ./base \
            --trained ./trained \
            --output report.html \
            --log-format json \
            > diagnostics.log

      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: afterburn-report
          path: |
            report.html
            diagnostics.log

      - name: Check for Reward Hacking
        run: |
          afterburn hack-check \
            --base ./base \
            --trained ./trained \
            --dataset gsm8k \
            --threshold 0.15 \
            --fail-on-detection
```

### GitLab CI Example

```yaml
afterburn:diagnostics:
  stage: validate
  image: python:3.11
  before_script:
    - pip install afterburn
  script:
    - afterburn diagnose --base $BASE_MODEL --trained $TRAINED_MODEL -o report.html
  artifacts:
    paths:
      - report.html
    expire_in: 30 days
  only:
    - main
```

## Performance Tuning

### CPU vs GPU

**GPU (Recommended):**
```bash
afterburn diagnose --base base --trained trained -o report.html
# Uses GPU automatically if available
```

**CPU-only:**
```bash
CUDA_VISIBLE_DEVICES="" afterburn diagnose --base base --trained trained -o report.html
# Slower but works without GPU
```

### Batch Size Tuning

For behavioral analysis, increase batch size for throughput:

```bash
afterburn behaviour \
  --base base \
  --trained trained \
  --dataset alpaca \
  --batch-size 32  # Default: 16, increase for faster inference
```

**Guidelines:**
- T4 / RTX 3060: batch_size = 8-16
- A10 / RTX 4090: batch_size = 16-32
- A100: batch_size = 32-64

### Skip Expensive Modules

Speed up analysis by skipping optional modules:

```bash
afterburn diagnose \
  --base base \
  --trained trained \
  --modules weight_diff behaviour  # Skip semantic, nli, etc.
  -o report.html
```

**Module runtime comparison (7B model):**
- `weight_diff`: ~2 minutes (always fast)
- `behaviour`: ~10 minutes (depends on dataset size)
- `reward_hack`: ~15 minutes (requires inference)
- `semantic`: ~20 minutes (requires embedding model)

### Disk I/O Optimization

Cache models on SSD and use local directories:

```bash
# Pre-download models
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir /ssd/models/base

# Use local paths
afterburn diagnose --base /ssd/models/base --trained /ssd/models/trained -o report.html
```

## Monitoring

### JSON Log Aggregation

Enable JSON logging for structured log ingestion:

```bash
afterburn diagnose \
  --base base \
  --trained trained \
  --log-format json \
  -o report.html \
  > afterburn.log 2>&1
```

**Sample JSON output:**
```json
{"timestamp": "2026-02-15T14:30:00", "level": "INFO", "logger": "afterburn.engine", "message": "Loading base model"}
{"timestamp": "2026-02-15T14:30:15", "level": "INFO", "logger": "afterburn.weight_diff", "message": "Processing layer 5/32"}
{"timestamp": "2026-02-15T14:31:00", "level": "WARNING", "logger": "afterburn.behaviour", "message": "High divergence detected in layer 12"}
```

### ELK Stack Integration

**Filebeat configuration:**
```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/afterburn/*.log
    json.keys_under_root: true
    json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "afterburn-logs-%{+yyyy.MM.dd}"
```

### CloudWatch Integration

```bash
aws logs create-log-group --log-group-name /afterburn/diagnostics

afterburn diagnose \
  --base base \
  --trained trained \
  --log-format json \
  2>&1 | aws logs put-log-events \
    --log-group-name /afterburn/diagnostics \
    --log-stream-name $(date +%Y%m%d-%H%M%S)
```

### Prometheus Metrics

Parse JSON logs to extract metrics:

```python
import json

def extract_metrics(log_file):
    metrics = {"layer_processing_time": [], "warnings": 0}
    with open(log_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry["level"] == "WARNING":
                metrics["warnings"] += 1
            if "Processing layer" in entry["message"]:
                # Extract timing info
                pass
    return metrics
```

## Troubleshooting

### Out of Memory (OOM)

**Symptom:** Process killed or CUDA OOM error

**Solutions:**
1. Use smaller batch size: `--batch-size 8`
2. Enable CPU offloading: `--device cpu`
3. Reduce model precision: Use quantized models (int8)
4. Skip memory-intensive modules: `--modules weight_diff`

**Example:**
```bash
# Low-memory mode
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
afterburn diagnose --base base --trained trained --batch-size 4 -o report.html
```

### Model Not Found

**Symptom:** `OSError: meta-llama/Llama-3.1-8B does not exist`

**Solutions:**
1. Authenticate with HuggingFace:
```bash
huggingface-cli login
# or
export HF_TOKEN=hf_xxxxx
```

2. Check model name spelling and access permissions
3. Pre-download model:
```bash
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir ./model
afterburn diagnose --base ./model --trained trained -o report.html
```

### HuggingFace Rate Limiting

**Symptom:** `HTTP 429: Too Many Requests`

**Solutions:**
1. Use HF_TOKEN for authenticated access (higher rate limits)
2. Cache models locally:
```bash
export HF_HOME=/persistent/cache
```
3. Add retry logic or wait before retrying

### Slow Analysis

**Symptom:** Diagnostics taking hours

**Quick fixes:**
1. Use GPU instead of CPU: Check `nvidia-smi`
2. Reduce dataset size: `--max-samples 100`
3. Skip slow modules: `--modules weight_diff behaviour`
4. Check disk I/O: Use SSD, avoid network mounts

**Profiling:**
```bash
# Enable verbose logging to identify bottlenecks
afterburn --verbose diagnose --base base --trained trained -o report.html
```

### Permission Denied (Docker)

**Symptom:** `PermissionError: [Errno 13] Permission denied`

**Solution:** Fix volume permissions:
```bash
# In docker-compose.yml, match user ID
user: "1000:1000"

# Or fix ownership
docker-compose run --rm afterburn chown -R 1000:1000 /app/reports
```

### Missing Dependencies

**Symptom:** `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution:** Install optional dependencies:
```bash
pip install afterburn[semantic]  # For semantic analysis
pip install afterburn[pdf]       # For PDF reports
pip install afterburn[nli]       # For NLI analysis
```

### CUDA Version Mismatch

**Symptom:** `CUDA driver version is insufficient`

**Solution:**
1. Check CUDA version: `nvidia-smi`
2. Install matching PyTorch:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Report Generation Fails

**Symptom:** Analysis completes but no report generated

**Debug:**
```bash
# Check write permissions
ls -la report.html

# Use absolute path
afterburn diagnose --base base --trained trained -o /tmp/report.html

# Check logs for errors
afterburn --verbose diagnose --base base --trained trained -o report.html
```
