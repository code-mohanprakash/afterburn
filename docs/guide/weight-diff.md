# Weight Diff Analysis

Compares model weights layer-by-layer without running inference.

## Metrics

| Metric | What It Measures |
|--------|-----------------|
| L2 Norm | Magnitude of weight change |
| Cosine Similarity | Direction preservation (1.0 = unchanged direction) |
| Frobenius Norm | Matrix-level change magnitude |
| Relative Change | Change normalized by original weight magnitude |
| SVD Decomposition | Effective rank, concentration ratio, stable rank of the diff |
| Spectral Alpha | Power-law exponent (2-4 = healthy, >6 = overfitting) |
| Marchenko-Pastur | Compares eigenvalue spectrum to random matrix theory |
| Behavioral Vectors | Principal directions of change via SVD, cross-layer coherence |
| Attention Heads | Per-head importance change |
| LayerNorm Shift | Gamma/beta parameter drift |
| Embedding Drift | Token embedding movement, most-drifted tokens |

## Memory Efficiency

Uses `safetensors.safe_open()` for memory-mapped weight access. Never loads both full models simultaneously. Peak memory is ~128MB per layer for 8B models.

## Usage

```python
from afterburn.weight_diff.engine import WeightDiffEngine
from afterburn.device import auto_detect_device
from afterburn.types import ModelPair

pair = ModelPair(base_model="base-model", trained_model="trained-model")
result = WeightDiffEngine(pair, auto_detect_device()).run()

for layer in result.top_changed_layers:
    print(f"{layer.layer_name}: relative_change={layer.relative_change:.4f}")
    if layer.mp_num_spikes is not None:
        print(f"  MP spikes: {layer.mp_num_spikes} (bulk: {layer.mp_bulk_fraction:.1%})")

print(f"Direction coherence: {result.direction_coherence:.3f}")
```
