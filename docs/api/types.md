# Types Reference

All types are frozen dataclasses defined in `afterburn.types`.

## Core Types

### ModelPair

```python
@dataclass(frozen=True)
class ModelPair:
    base_model: str          # HuggingFace ID or local path
    trained_model: str       # HuggingFace ID or local path
    method: TrainingMethod   # Training method used
```

### DiagnosticReport

Top-level report container. See [Diagnoser API](diagnoser.md#diagnosticreport).

## Weight Diff Types

### LayerDiff

Per-layer weight comparison metrics.

| Field | Type | Description |
|-------|------|-------------|
| `layer_name` | `str` | Layer identifier (e.g. `layer_5`) |
| `layer_index` | `int` | Numeric layer index |
| `l2_norm` | `float` | L2 norm of weight difference |
| `cosine_similarity` | `float` | Cosine similarity (1.0 = unchanged) |
| `frobenius_norm` | `float` | Frobenius norm of difference |
| `relative_change` | `float` | Change relative to base magnitude |
| `param_count` | `int` | Number of parameters in this layer |
| `svd_top_singular_values` | `tuple[float, ...] \| None` | Top singular values of the diff |
| `svd_effective_rank` | `int \| None` | Effective rank (90% energy) |
| `svd_concentration_ratio` | `float \| None` | Energy in top-1 SV / total |
| `svd_stable_rank` | `float \| None` | Frobenius^2 / spectral^2 |
| `spectral_alpha` | `float \| None` | Power-law exponent (2-4 = healthy) |
| `spectral_alpha_quality` | `str \| None` | Quality assessment: good/fair/poor/unstable |
| `mp_sigma_sq` | `float \| None` | Marchenko-Pastur noise variance |
| `mp_num_spikes` | `int \| None` | Eigenvalues above MP upper edge |
| `mp_bulk_fraction` | `float \| None` | Fraction within MP bulk |
| `mp_kl_divergence` | `float \| None` | KL divergence from theoretical MP |

### AttentionHeadScore

| Field | Type | Description |
|-------|------|-------------|
| `layer_index` | `int` | Layer number |
| `head_index` | `int` | Head number |
| `base_importance` | `float` | Base model head importance |
| `trained_importance` | `float` | Trained model head importance |
| `importance_delta` | `float` | Change in importance |

### LayerNormShift

| Field | Type | Description |
|-------|------|-------------|
| `layer_name` | `str` | Layer identifier |
| `layer_index` | `int` | Numeric layer index |
| `gamma_shift` | `float` | Mean absolute change in scale (gamma) |
| `beta_shift` | `float` | Mean absolute change in bias (beta) |
| `total_shift` | `float` | Combined relative shift |
| `is_significant` | `bool` | Whether shift exceeds threshold |

### EmbeddingDrift

| Field | Type | Description |
|-------|------|-------------|
| `input_embedding_l2` | `float` | L2 distance of input embeddings |
| `input_embedding_cosine` | `float` | Cosine similarity of input embeddings |
| `output_embedding_l2` | `float \| None` | L2 distance of output embeddings |
| `output_embedding_cosine` | `float \| None` | Cosine similarity of output embeddings |
| `top_drifted_tokens` | `list[tuple[int, float]]` | Most-changed token IDs with drift magnitude |

### BehavioralVector

| Field | Type | Description |
|-------|------|-------------|
| `layer_name` | `str` | Layer identifier |
| `singular_value` | `float` | Magnitude of this direction |
| `direction_index` | `int` | Rank (0 = dominant) |
| `explained_variance_ratio` | `float` | Fraction of total variance explained |

### WeightDiffResult

| Field | Type | Description |
|-------|------|-------------|
| `layer_diffs` | `list[LayerDiff]` | All layer diffs |
| `attention_heads` | `list[AttentionHeadScore]` | Per-head scores |
| `layernorm_shifts` | `list[LayerNormShift]` | LayerNorm parameter shifts |
| `embedding_drift` | `EmbeddingDrift` | Embedding layer movement |
| `lora_analysis` | `dict \| None` | LoRA adapter analysis (if detected) |
| `total_param_count` | `int` | Total parameters compared |
| `changed_param_count` | `int` | Parameters with measurable change |
| `behavioral_vectors` | `list[BehavioralVector]` | Principal change directions |
| `direction_coherence` | `float` | Cross-layer direction alignment (0-1) |

**Properties:** `top_changed_layers`, `change_concentration`

## Behaviour Types

### LengthAnalysis

| Field | Type | Description |
|-------|------|-------------|
| `base_mean` | `float` | Mean output length (base model) |
| `trained_mean` | `float` | Mean output length (trained model) |
| `mean_diff` | `float` | Difference in means |
| `cohens_d` | `float` | Effect size (0.2=small, 0.5=medium, 0.8=large) |
| `p_value` | `float` | Mann-Whitney U test p-value |
| `is_significant` | `bool` | p < 0.05 and \|d\| > 0.2 |

### FormatAnalysis

| Field | Type | Description |
|-------|------|-------------|
| `base_format_rate` | `float` | Format pattern match rate (base) |
| `trained_format_rate` | `float` | Format pattern match rate (trained) |
| `format_increase` | `float` | Rate increase |
| `patterns_detected` | `dict[str, dict]` | Per-pattern base/trained rates |

### StrategyShiftAnalysis

| Field | Type | Description |
|-------|------|-------------|
| `base_distribution` | `dict[str, float]` | Strategy frequency distribution (base) |
| `trained_distribution` | `dict[str, float]` | Strategy frequency distribution (trained) |
| `dominant_shift` | `str` | Most prominent strategy change |
| `base_entropy` | `float` | Shannon entropy of base distribution |
| `trained_entropy` | `float` | Shannon entropy of trained distribution |
| `entropy_change` | `float` | Entropy difference |

### ChainOfThoughtAnalysis

| Field | Type | Description |
|-------|------|-------------|
| `base_avg_steps` | `float` | Average reasoning steps (base) |
| `trained_avg_steps` | `float` | Average reasoning steps (trained) |
| `step_count_change` | `float` | Change in step count |
| `base_avg_depth` | `float` | Average reasoning depth (base) |
| `trained_avg_depth` | `float` | Average reasoning depth (trained) |
| `depth_change` | `float` | Change in depth |

### DiversityAnalysis

| Field | Type | Description |
|-------|------|-------------|
| `base_ead` | `dict[int, float]` | EAD scores per n-gram level (base) |
| `trained_ead` | `dict[int, float]` | EAD scores per n-gram level (trained) |
| `base_diversity_score` | `float` | Mean EAD across n=1..5 (base) |
| `trained_diversity_score` | `float` | Mean EAD across n=1..5 (trained) |
| `diversity_change` | `float` | Change in diversity score |

### TokenDivergenceAnalysis

| Field | Type | Description |
|-------|------|-------------|
| `mean_jsd` | `float` | Mean Jensen-Shannon Divergence [0, 1] |
| `per_prompt_jsd` | `list[float]` | JSD values per prompt |
| `top_divergent_prompts` | `list[tuple[str, float]]` | Most divergent (prompt_id, JSD) |
| `has_token_probs` | `bool` | Whether token probability data was available |

## Reward Hack Types

### RewardHackResult

| Field | Type | Description |
|-------|------|-------------|
| `composite_score` | `float` | Weighted risk score (0-100) |
| `risk_level` | `RiskLevel` | LOW / MODERATE / HIGH / CRITICAL |
| `flags` | `list[str]` | Human-readable warning flags |
| `length_bias` | `LengthBiasResult` | Length bias detection result |
| `format_gaming` | `FormatGamingResult` | Format gaming detection result |
| `strategy_collapse` | `StrategyCollapseResult` | Strategy collapse result |
| `sycophancy` | `SycophancyResult` | Sycophancy detection result |

### Sub-detector Results

Each sub-detector result has at minimum:

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Individual detector score (0-100) |
| `is_flagged` | `bool` | Whether this detector triggered |
| `detail` | `str` | Human-readable explanation |

## Enums

| Enum | Values |
|------|--------|
| `TrainingMethod` | `sft`, `dpo`, `rlhf`, `rlvr`, `grpo`, `lora`, `qlora`, `unknown` |
| `ReportFormat` | `html`, `markdown`, `pdf`, `json` |
| `RiskLevel` | `low`, `moderate`, `high`, `critical` |
| `ReasoningStrategy` | `direct_answer`, `step_by_step`, `code_assisted`, `chain_of_thought`, `tool_use`, `unknown` |
