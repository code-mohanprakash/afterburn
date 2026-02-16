# Diagnoser API

The `Diagnoser` class is the main entry point for running diagnostics.

## Constructor

```python
from afterburn import Diagnoser

diag = Diagnoser(
    base_model="meta-llama/Llama-3.1-8B",
    trained_model="my-org/Llama-3.1-8B-SFT",
    method="sft",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | `str` | *required* | HuggingFace model ID or local path to the base model |
| `trained_model` | `str` | *required* | HuggingFace model ID or local path to the trained model |
| `method` | `str` | `"unknown"` | Training method: `sft`, `dpo`, `rlhf`, `rlvr`, `grpo`, `lora`, `qlora` |
| `suites` | `list[str]` | `None` | Prompt suites to run. Default: `["math", "code", "reasoning", "safety"]` |
| `config_path` | `str` | `None` | Path to `.afterburn.yaml` configuration file |
| `device` | `str` | `None` | Force device: `cuda`, `mps`, `cpu`. Default: auto-detect |
| `modules` | `list[str]` | `None` | Modules to run: `weight_diff`, `behaviour`, `reward_hack`. Default: all |
| `collect_logits` | `bool` | `False` | Collect token-level probabilities for JSD analysis |

## Methods

### `run() -> DiagnosticReport`

Run full diagnostic analysis (weight diff + behaviour + reward hack).

```python
report = diag.run()
print(report.summary)
print(f"Risk score: {report.hack_score:.0f}/100")
report.save("report.html")
```

### `run_weight_diff() -> WeightDiffResult`

Run weight diff analysis only. No inference needed — compares weights directly.

```python
wd = diag.run_weight_diff()
for layer in wd.top_changed_layers:
    print(f"{layer.layer_name}: {layer.relative_change:.4f}")
```

### `run_behaviour() -> BehaviourResult`

Run behavioural analysis only. Loads each model sequentially for inference.

```python
bh = diag.run_behaviour()
print(f"Length change: {bh.length_analysis.mean_diff:+.1f} tokens")
print(f"Strategy shift: {bh.strategy_analysis.dominant_shift}")
```

### `run_hack_check() -> RewardHackResult`

Run reward hack detection. Requires behaviour results (runs inference if needed).

```python
rh = diag.run_hack_check()
print(f"Composite score: {rh.composite_score:.0f}/100 ({rh.risk_level.value})")
for flag in rh.flags:
    print(f"  - {flag}")
```

## DiagnosticReport

The top-level report container returned by `run()`.

| Field | Type | Description |
|-------|------|-------------|
| `model_pair` | `ModelPair` | Base and trained model identifiers |
| `weight_diff` | `WeightDiffResult \| None` | Weight diff results (if run) |
| `behaviour` | `BehaviourResult \| None` | Behaviour results (if run) |
| `reward_hack` | `RewardHackResult \| None` | Reward hack results (if run) |
| `summary` | `str` | Plain-English executive summary |
| `hack_score` | `float` | Composite reward hack score (0-100) |
| `recommendations` | `list[str]` | Actionable recommendations |

### Methods

| Method | Description |
|--------|-------------|
| `save(path)` | Save report to file (format auto-detected from extension) |
| `to_json()` | Return report as JSON-serializable dict |

## Lower-Level APIs

### WeightDiffEngine

For direct weight comparison without the orchestrator:

```python
from afterburn.weight_diff.engine import WeightDiffEngine
from afterburn.device import auto_detect_device
from afterburn.types import ModelPair

pair = ModelPair(base_model="base-path", trained_model="trained-path")
engine = WeightDiffEngine(pair, auto_detect_device())
result = engine.run()
```

### BehaviourAnalyser

```python
from afterburn.behaviour.analyser import BehaviourAnalyser
from afterburn.device import auto_detect_device
from afterburn.types import ModelPair

pair = ModelPair(base_model="base-path", trained_model="trained-path")
analyser = BehaviourAnalyser(pair, auto_detect_device(), suites=["math"])
result = analyser.run()
```

### RewardHackDetector

```python
from afterburn.reward_hack.detector import RewardHackDetector

detector = RewardHackDetector()
result = detector.detect(behaviour_result, training_method="sft")
```
