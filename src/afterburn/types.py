"""Shared types, dataclasses, and enums for Afterburn."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfidenceInterval:
    """A confidence interval with lower/upper bounds and confidence level."""

    lower: float
    upper: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.upper - self.lower


class TrainingMethod(Enum):
    SFT = "sft"
    DPO = "dpo"
    RLHF = "rlhf"
    RLVR = "rlvr"
    GRPO = "grpo"
    LORA = "lora"
    QLORA = "qlora"
    UNKNOWN = "unknown"


class ReportFormat(Enum):
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"


class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> RiskLevel:
        if score <= 25:
            return cls.LOW
        elif score <= 50:
            return cls.MODERATE
        elif score <= 75:
            return cls.HIGH
        else:
            return cls.CRITICAL


class ReasoningStrategy(Enum):
    DIRECT_ANSWER = "direct_answer"
    STEP_BY_STEP = "step_by_step"
    CODE_ASSISTED = "code_assisted"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TOOL_USE = "tool_use"
    UNKNOWN = "unknown"


# ─── Weight Diff Types ───────────────────────────────────────────────


@dataclass(frozen=True)
class LayerDiff:
    """Weight difference metrics for a single layer."""

    layer_name: str
    layer_index: int
    l2_norm: float
    cosine_similarity: float
    frobenius_norm: float
    relative_change: float
    param_count: int
    # SVD analysis of weight diff (None for 1D/bias layers)
    svd_top_singular_values: tuple[float, ...] | None = None
    svd_effective_rank: int | None = None
    svd_concentration_ratio: float | None = None
    svd_stable_rank: float | None = None
    # Spectral analysis of the trained weight matrix
    spectral_alpha: float | None = None
    spectral_alpha_quality: str | None = None
    spectral_stable_rank: float | None = None
    # Marchenko-Pastur law comparison
    mp_sigma_sq: float | None = None
    mp_num_spikes: int | None = None
    mp_bulk_fraction: float | None = None
    mp_kl_divergence: float | None = None


@dataclass(frozen=True)
class AttentionHeadScore:
    """Importance score for a single attention head."""

    layer_index: int
    head_index: int
    base_importance: float
    trained_importance: float
    importance_delta: float


@dataclass(frozen=True)
class LayerNormShift:
    """Detected shift in LayerNorm parameters."""

    layer_name: str
    layer_index: int
    gamma_shift: float
    beta_shift: float
    total_shift: float
    is_significant: bool


@dataclass(frozen=True)
class EmbeddingDrift:
    """Embedding layer drift measurement."""

    input_embedding_l2: float
    input_embedding_cosine: float
    output_embedding_l2: float | None
    output_embedding_cosine: float | None
    top_drifted_tokens: list[tuple[int, float]]


@dataclass(frozen=True)
class BehavioralVector:
    """A principal direction of change extracted from weight diff SVD."""
    layer_name: str
    singular_value: float
    direction_index: int          # 0 = top, 1 = second, etc.
    explained_variance_ratio: float  # SV^2 / sum(SV^2)


# ─── Prompt Types ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Prompt:
    """A single prompt in a suite."""

    id: str
    text: str
    category: str
    expected_answer: str | None = None
    difficulty: str | None = None
    tags: tuple[str, ...] = ()


@dataclass
class PromptResult:
    """Result from running a single prompt through a model."""

    prompt_id: str
    prompt_text: str
    category: str
    output_text: str
    output_tokens: int
    generation_time_ms: float
    expected_answer: str | None = None
    top_token_probs: list[dict[str, float]] | None = None


# ─── Behaviour Types ─────────────────────────────────────────────────


@dataclass
class LengthAnalysis:
    """Output length distribution comparison."""

    base_mean: float
    base_median: float
    base_std: float
    trained_mean: float
    trained_median: float
    trained_std: float
    mean_diff: float
    p_value: float
    cohens_d: float
    is_significant: bool
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    base_skewness: float = 0.0
    trained_skewness: float = 0.0
    base_kurtosis: float = 0.0
    trained_kurtosis: float = 0.0
    base_percentiles: dict[str, float] = field(default_factory=dict)
    trained_percentiles: dict[str, float] = field(default_factory=dict)
    cohens_d_ci: ConfidenceInterval | None = None
    mean_diff_ci: ConfidenceInterval | None = None
    corrected_p_value: float | None = None


@dataclass
class FormatAnalysis:
    """Format compliance scoring."""

    patterns_detected: dict[str, dict[str, float]] = field(default_factory=dict)
    base_format_rate: float = 0.0
    trained_format_rate: float = 0.0
    format_increase: float = 0.0
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    base_format_diversity: float = 0.0
    trained_format_diversity: float = 0.0


@dataclass
class StrategyShiftAnalysis:
    """Reasoning strategy shift analysis."""

    base_distribution: dict[str, float] = field(default_factory=dict)
    trained_distribution: dict[str, float] = field(default_factory=dict)
    dominant_shift: str = ""
    base_entropy: float = 0.0
    trained_entropy: float = 0.0
    entropy_change: float = 0.0


@dataclass
class ChainOfThoughtAnalysis:
    """Chain-of-thought pattern analysis."""

    base_avg_steps: float = 0.0
    trained_avg_steps: float = 0.0
    base_avg_depth: float = 0.0
    trained_avg_depth: float = 0.0
    step_count_change: float = 0.0
    depth_change: float = 0.0
    base_self_correction_rate: float = 0.0
    trained_self_correction_rate: float = 0.0
    base_verification_rate: float = 0.0
    trained_verification_rate: float = 0.0


@dataclass(frozen=True)
class CalibrationBin:
    """A single bin in a calibration reliability diagram."""

    bin_lower: float
    bin_upper: float
    avg_confidence: float
    avg_accuracy: float
    count: int


@dataclass
class CalibrationAnalysis:
    """Confidence calibration comparison."""

    base_ece: float = 0.0
    trained_ece: float = 0.0
    calibration_change: float = 0.0
    base_bins: list[CalibrationBin] = field(default_factory=list)
    trained_bins: list[CalibrationBin] = field(default_factory=list)
    base_overconfidence_rate: float = 0.0
    trained_overconfidence_rate: float = 0.0
    has_token_probs: bool = False


# ─── Reward Hack Types ───────────────────────────────────────────────


@dataclass
class LengthBiasResult:
    """Length bias detection result."""

    score: float
    cohens_d: float
    p_value: float
    mean_length_ratio: float
    is_flagged: bool
    detail: str = ""
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    cohens_d_ci: ConfidenceInterval | None = None
    corrected_p_value: float | None = None


@dataclass
class FormatGamingResult:
    """Format gaming detection result."""

    score: float
    patterns: dict[str, dict[str, float]] = field(default_factory=dict)
    is_flagged: bool = False
    detail: str = ""


@dataclass
class StrategyCollapseResult:
    """Strategy collapse detection result."""

    score: float
    base_entropy: float
    trained_entropy: float
    entropy_drop: float
    dominant_strategy: str = ""
    is_flagged: bool = False
    detail: str = ""
    entropy_drop_ci: ConfidenceInterval | None = None


@dataclass
class SycophancyResult:
    """Sycophancy detection result."""

    score: float
    base_agreement_rate: float = 0.0
    trained_agreement_rate: float = 0.0
    agreement_increase: float = 0.0
    is_flagged: bool = False
    detail: str = ""
    base_pushback_rate: float = 0.0
    trained_pushback_rate: float = 0.0
    persuasion_resistance_drop: float = 0.0
    num_probes_used: int = 0
    per_domain_consistency: dict[str, float] = field(default_factory=dict)
    agreement_rate_ci: ConfidenceInterval | None = None
    pushback_rate_ci: ConfidenceInterval | None = None


# ─── Aggregate Result Types ──────────────────────────────────────────


@dataclass
class WeightDiffResult:
    """Complete weight diff analysis results."""

    layer_diffs: list[LayerDiff]
    attention_heads: list[AttentionHeadScore]
    layernorm_shifts: list[LayerNormShift]
    embedding_drift: EmbeddingDrift
    lora_analysis: dict[str, Any] | None
    total_param_count: int
    changed_param_count: int
    behavioral_vectors: list[BehavioralVector] = field(default_factory=list)
    direction_coherence: float = 0.0

    @property
    def top_changed_layers(self) -> list[LayerDiff]:
        return sorted(self.layer_diffs, key=lambda x: x.relative_change, reverse=True)[:5]

    @property
    def change_concentration(self) -> float:
        if not self.layer_diffs:
            return 0.0
        total = sum(d.frobenius_norm for d in self.layer_diffs)
        if total < 1e-10:
            return 0.0
        top5 = sum(d.frobenius_norm for d in self.top_changed_layers)
        return top5 / total


@dataclass
class TokenDivergenceAnalysis:
    """Token-level probability divergence between base and trained models."""

    mean_jsd: float = 0.0
    per_prompt_jsd: list[float] = field(default_factory=list)
    top_divergent_prompts: list[tuple[str, float]] = field(default_factory=list)
    per_category: dict[str, float] = field(default_factory=dict)
    has_token_probs: bool = False
    num_prompts_analyzed: int = 0
    mean_jsd_ci: ConfidenceInterval | None = None


@dataclass
class DiversityAnalysis:
    """Output diversity comparison using EAD and optional SBERT semantic similarity."""

    base_ead: dict[int, float] = field(default_factory=dict)
    trained_ead: dict[int, float] = field(default_factory=dict)
    base_diversity_score: float = 0.0
    trained_diversity_score: float = 0.0
    diversity_change: float = 0.0
    base_semantic_diversity: float | None = None
    trained_semantic_diversity: float | None = None
    semantic_diversity_change: float | None = None
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class BehaviourResult:
    """Complete behaviour analysis results."""

    base_results: list[PromptResult]
    trained_results: list[PromptResult]
    length_analysis: LengthAnalysis
    format_analysis: FormatAnalysis
    strategy_analysis: StrategyShiftAnalysis
    cot_analysis: ChainOfThoughtAnalysis
    calibration: CalibrationAnalysis | None = None
    diversity: DiversityAnalysis | None = None
    token_divergence: TokenDivergenceAnalysis | None = None

    @property
    def summary(self) -> str:
        parts = []
        if self.length_analysis.is_significant:
            direction = "longer" if self.length_analysis.mean_diff > 0 else "shorter"
            parts.append(
                f"Outputs are significantly {direction} "
                f"(mean diff: {self.length_analysis.mean_diff:+.1f} tokens, "
                f"p={self.length_analysis.p_value:.4f})"
            )
        if self.strategy_analysis.dominant_shift:
            parts.append(f"Reasoning strategy shift: {self.strategy_analysis.dominant_shift}")
        if self.format_analysis.format_increase > 0.1:
            parts.append(
                f"Format compliance increased by "
                f"{self.format_analysis.format_increase:.1%}"
            )
        return ". ".join(parts) if parts else "No significant behavioural changes detected."


@dataclass
class RewardHackResult:
    """Complete reward hacking analysis results."""

    length_bias: LengthBiasResult
    format_gaming: FormatGamingResult
    strategy_collapse: StrategyCollapseResult
    sycophancy: SycophancyResult
    composite_score: float
    risk_level: RiskLevel
    flags: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.flags:
            return (
                f"Reward hacking risk: {self.risk_level.value.upper()} "
                f"({self.composite_score:.0f}/100). No specific flags raised."
            )
        flag_text = "; ".join(self.flags)
        return (
            f"Reward hacking risk: {self.risk_level.value.upper()} "
            f"({self.composite_score:.0f}/100). Flags: {flag_text}"
        )


# ─── Top-Level Types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelPair:
    """Identifies the two models being compared."""

    base_model: str
    trained_model: str
    method: TrainingMethod = TrainingMethod.UNKNOWN


@dataclass
class DiagnosticReport:
    """Complete diagnostic report aggregating all analysis results."""

    model_pair: ModelPair
    weight_diff: WeightDiffResult | None = None
    behaviour: BehaviourResult | None = None
    reward_hack: RewardHackResult | None = None
    summary: str = ""
    hack_score: float = 0.0
    top_changed_layers: list[LayerDiff] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path, fmt: ReportFormat | None = None) -> Path:
        from afterburn.report.generator import ReportGenerator

        path = Path(path)
        if fmt is None:
            fmt = _detect_format(path)
        generator = ReportGenerator(self)
        return generator.generate(fmt, path)


def _detect_format(path: Path) -> ReportFormat:
    suffix_map = {
        ".html": ReportFormat.HTML,
        ".htm": ReportFormat.HTML,
        ".md": ReportFormat.MARKDOWN,
        ".pdf": ReportFormat.PDF,
        ".json": ReportFormat.JSON,
    }
    fmt = suffix_map.get(path.suffix.lower())
    if fmt is None:
        return ReportFormat.HTML
    return fmt
