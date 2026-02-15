"""Tests for reasoning strategy classification."""

import pytest

from afterburn.behaviour.reasoning import classify_reasoning_strategy
from afterburn.types import ReasoningStrategy


class TestClassifyReasoningStrategy:
    def test_code_assisted(self):
        output = """
        ```python
        def solve(x):
            return x * 2
        result = solve(5)
        ```
        """
        assert classify_reasoning_strategy(output) == ReasoningStrategy.CODE_ASSISTED

    def test_step_by_step(self):
        output = """
        Step 1: Identify the variables
        Step 2: Set up the equation
        Step 3: Solve for x
        """
        assert classify_reasoning_strategy(output) == ReasoningStrategy.STEP_BY_STEP

    def test_chain_of_thought(self):
        output = """
        Let me think about this carefully. I need to consider the problem.
        So, if we look at it this way, the answer becomes clearer.
        """
        assert classify_reasoning_strategy(output) == ReasoningStrategy.CHAIN_OF_THOUGHT

    def test_direct_answer(self):
        assert classify_reasoning_strategy("42") == ReasoningStrategy.DIRECT_ANSWER
        assert classify_reasoning_strategy("The answer is 5.") == ReasoningStrategy.DIRECT_ANSWER

    def test_empty_string(self):
        assert classify_reasoning_strategy("") == ReasoningStrategy.UNKNOWN

    def test_numbered_list(self):
        output = """
        1. First we need to find x
        2. Then we substitute
        3. Finally we solve
        """
        assert classify_reasoning_strategy(output) == ReasoningStrategy.STEP_BY_STEP

    def test_code_with_imports(self):
        output = """
        import math
        def calculate(n):
            return math.sqrt(n)
        result = calculate(16)
        """
        assert classify_reasoning_strategy(output) == ReasoningStrategy.CODE_ASSISTED
