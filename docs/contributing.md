# Contributing

## Development Setup

```bash
git clone https://github.com/code-mohanprakash/afterburn.git
cd afterburn
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=afterburn --cov-report=term-missing
```

## Code Style

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Type Checking

```bash
mypy src/afterburn/ --ignore-missing-imports
```

## Architecture

```
src/afterburn/
├── loading/        # Model loading, safetensors, LoRA
├── weight_diff/    # Layer-by-layer weight comparison
├── behaviour/      # Behavioral shift analysis
├── reward_hack/    # Reward hacking detection
├── prompts/        # Prompt suites + inference runner
├── report/         # HTML/JSON/MD/PDF report generation
├── nli.py          # Shared NLI model (cross-encoder)
├── diagnoser.py    # Top-level orchestrator
└── types.py        # Shared dataclasses and enums
```

## Adding a Reward Hack Detector

1. Create `src/afterburn/reward_hack/my_detector.py`
2. Return a result dataclass (add to `types.py`)
3. Wire into `detector.py` orchestrator
4. Add to `risk_score.py` composite calculation
5. Add tests in `tests/test_reward_hack/`

## Adding a Prompt Suite

Create a YAML file:

```yaml
name: "my-domain"
category: "custom"
prompts:
  - id: "q1"
    text: "Your prompt"
    expected_answer: "Expected answer"
    difficulty: "medium"
    tags: ["domain"]
```
