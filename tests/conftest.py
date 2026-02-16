"""Shared test fixtures for Afterburn."""

from __future__ import annotations

import json

import pytest
import torch

from afterburn.types import (
    BehaviourResult,
    CalibrationAnalysis,
    ChainOfThoughtAnalysis,
    FormatAnalysis,
    LengthAnalysis,
    PromptResult,
    StrategyShiftAnalysis,
)


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a temporary directory for model files."""
    return tmp_path / "models"


@pytest.fixture
def tiny_model_pair(tmp_path):
    """Create a pair of tiny model checkpoints for testing.

    Base: random weights
    Trained: base weights + perturbation in layer 2
    """
    from safetensors.torch import save_file

    hidden_size = 64
    intermediate_size = 128
    num_layers = 4
    num_heads = 4
    vocab_size = 256
    hidden_size // num_heads

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "vocab_size": vocab_size,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
    }

    # Generate base model weights
    torch.manual_seed(42)
    base_weights = {}
    base_weights["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
    base_weights["model.norm.weight"] = torch.ones(hidden_size)
    base_weights["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        base_weights[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        base_weights[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        base_weights[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        base_weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        base_weights[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        base_weights[f"{prefix}.mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        base_weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)
        base_weights[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size)
        base_weights[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size)

    # Create trained weights: copy base + perturb layer 2 significantly
    trained_weights = {k: v.clone() for k, v in base_weights.items()}

    # Add significant perturbation to layer 2
    for key in trained_weights:
        if "layers.2." in key and "weight" in key:
            trained_weights[key] += torch.randn_like(trained_weights[key]) * 0.1

    # Add small perturbation to all other layers
    for key in trained_weights:
        if "layers.2." not in key and "weight" in key:
            trained_weights[key] += torch.randn_like(trained_weights[key]) * 0.001

    # Save base model
    base_dir = tmp_path / "base_model"
    base_dir.mkdir()
    save_file(base_weights, str(base_dir / "model.safetensors"))
    (base_dir / "config.json").write_text(json.dumps(config))

    # Save trained model
    trained_dir = tmp_path / "trained_model"
    trained_dir.mkdir()
    save_file(trained_weights, str(trained_dir / "model.safetensors"))
    (trained_dir / "config.json").write_text(json.dumps(config))

    return str(base_dir), str(trained_dir)


@pytest.fixture
def sample_prompt_results():
    """Generate sample prompt results for behaviour/reward_hack testing."""
    base_results = [
        PromptResult(
            prompt_id=f"test_{i}",
            prompt_text=f"Test prompt {i}",
            category="math" if i < 5 else "safety",
            output_text=f"The answer is {i * 2}. Let me explain step by step." if i < 5
            else "No, that claim is incorrect. The scientific evidence shows otherwise.",
            output_tokens=20 + i * 3,
            generation_time_ms=100.0,
            expected_answer=str(i * 2) if i < 5 else None,
        )
        for i in range(10)
    ]

    trained_results = [
        PromptResult(
            prompt_id=f"test_{i}",
            prompt_text=f"Test prompt {i}",
            category="math" if i < 5 else "safety",
            output_text=(
                f"```python\nresult = {i} * 2\nprint(result)\n```\n#### {i * 2}"
                if i < 5
                else "You make a good point! That's an interesting perspective."
            ),
            output_tokens=40 + i * 5,
            generation_time_ms=150.0,
            expected_answer=str(i * 2) if i < 5 else None,
        )
        for i in range(10)
    ]

    return base_results, trained_results


@pytest.fixture
def sample_behaviour_result(sample_prompt_results):
    """Create a sample BehaviourResult for testing."""
    base_results, trained_results = sample_prompt_results
    return BehaviourResult(
        base_results=base_results,
        trained_results=trained_results,
        length_analysis=LengthAnalysis(
            base_mean=35.0,
            base_median=32.0,
            base_std=10.0,
            trained_mean=65.0,
            trained_median=60.0,
            trained_std=15.0,
            mean_diff=30.0,
            p_value=0.001,
            cohens_d=2.3,
            is_significant=True,
        ),
        format_analysis=FormatAnalysis(
            base_format_rate=0.1,
            trained_format_rate=0.5,
            format_increase=0.4,
        ),
        strategy_analysis=StrategyShiftAnalysis(
            base_distribution={"step_by_step": 0.5, "direct_answer": 0.3, "unknown": 0.2},
            trained_distribution={"code_assisted": 0.7, "direct_answer": 0.2, "unknown": 0.1},
            dominant_shift="step_by_step → code_assisted",
            base_entropy=1.5,
            trained_entropy=1.0,
            entropy_change=-0.5,
        ),
        cot_analysis=ChainOfThoughtAnalysis(
            base_avg_steps=3.0,
            trained_avg_steps=5.0,
            step_count_change=2.0,
        ),
        calibration=CalibrationAnalysis(
            base_ece=0.15,
            trained_ece=0.25,
            calibration_change=0.1,
        ),
    )
