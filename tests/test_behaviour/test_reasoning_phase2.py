"""Tests for reasoning strategy classifier (simplified priority-based)."""

import pytest

from afterburn.behaviour.reasoning import (
    ClassificationResult,
    analyze_strategy_shift,
    classify_reasoning_strategy,
    classify_reasoning_strategy_detailed,
)
from afterburn.types import PromptResult, ReasoningStrategy


class TestDetailedClassification:
    def test_returns_classification_result(self):
        result = classify_reasoning_strategy_detailed("The answer is 42.")
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.strategy, ReasoningStrategy)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.scores, dict)

    def test_empty_string_is_unknown(self):
        result = classify_reasoning_strategy_detailed("")
        assert result.strategy == ReasoningStrategy.UNKNOWN
        assert result.confidence == 1.0

    def test_whitespace_only_is_unknown(self):
        result = classify_reasoning_strategy_detailed("   \n  ")
        assert result.strategy == ReasoningStrategy.UNKNOWN


class TestToolUseDetection:
    def test_bash_code_block(self):
        output = "```bash\ncurl https://api.example.com/data\nwget https://example.com\n```"
        assert classify_reasoning_strategy(output) == ReasoningStrategy.TOOL_USE

    def test_function_call_keyword(self):
        output = "Use function_call to invoke tool_use for the latest information."
        result = classify_reasoning_strategy_detailed(output)
        assert result.scores["tool_use"] > 0


class TestCodeAssisted:
    def test_python_code_block(self):
        output = """Here's how to solve it:
```python
def solve(x, y):
    return x + y
result = solve(3, 4)
print(result)
```
The answer is 7."""
        assert classify_reasoning_strategy(output) == ReasoningStrategy.CODE_ASSISTED

    def test_code_with_imports(self):
        output = "import numpy as np\ndef compute():\n    return np.sqrt(16)\nresult = compute()"
        assert classify_reasoning_strategy(output) == ReasoningStrategy.CODE_ASSISTED


class TestStepByStep:
    def test_explicit_steps(self):
        output = "Step 1: Identify the variables.\nStep 2: Set up the equation.\nStep 3: Solve."
        assert classify_reasoning_strategy(output) == ReasoningStrategy.STEP_BY_STEP

    def test_numbered_list(self):
        output = "1. First we need to find x\n2. Then we substitute\n3. Finally we solve"
        assert classify_reasoning_strategy(output) == ReasoningStrategy.STEP_BY_STEP


class TestChainOfThought:
    def test_exploratory_reasoning(self):
        output = """Let me think about this step by step.
I need to first consider what we know about the problem.
So, if we apply the formula, we get a certain result.
Wait, actually I should reconsider this approach.
Let's verify our answer by checking the boundary conditions."""
        assert classify_reasoning_strategy(output) == ReasoningStrategy.CHAIN_OF_THOUGHT


class TestDirectAnswer:
    def test_short_answer(self):
        assert classify_reasoning_strategy("42") == ReasoningStrategy.DIRECT_ANSWER

    def test_short_sentence(self):
        assert classify_reasoning_strategy("The answer is 5.") == ReasoningStrategy.DIRECT_ANSWER


class TestStrategyShiftAnalysis:
    def test_detects_shift(self):
        base = [
            PromptResult(
                prompt_id="p1", prompt_text="test", category="math",
                output_text="Step 1: Start. Step 2: Calculate. Step 3: Answer is 42.",
                output_tokens=20, generation_time_ms=100.0,
            )
            for _ in range(5)
        ]
        trained = [
            PromptResult(
                prompt_id="p1", prompt_text="test", category="math",
                output_text="```python\nresult = 42\nprint(result)\n```\nThe answer is 42.",
                output_tokens=20, generation_time_ms=100.0,
            )
            for _ in range(5)
        ]
        analysis = analyze_strategy_shift(base, trained)
        assert analysis.base_distribution != analysis.trained_distribution
        assert analysis.dominant_shift != ""

    def test_empty_inputs(self):
        analysis = analyze_strategy_shift([], [])
        assert analysis.dominant_shift == ""
        assert analysis.base_entropy == 0.0
