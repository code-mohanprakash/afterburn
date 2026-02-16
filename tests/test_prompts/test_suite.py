"""Tests for prompt suite loading."""

import pytest
import yaml

from afterburn.exceptions import PromptSuiteError
from afterburn.prompts.suite import BUILTIN_SUITES, PromptSuite


class TestPromptSuite:
    def test_load_builtin_math(self):
        suite = PromptSuite.builtin("math")
        assert suite.name == "math-reasoning"
        assert suite.category == "math"
        assert len(suite.prompts) > 0

    def test_load_builtin_code(self):
        suite = PromptSuite.builtin("code")
        assert len(suite.prompts) > 0
        assert suite.category == "code"

    def test_load_builtin_reasoning(self):
        suite = PromptSuite.builtin("reasoning")
        assert len(suite.prompts) > 0

    def test_load_builtin_safety(self):
        suite = PromptSuite.builtin("safety")
        assert len(suite.prompts) > 0
        assert suite.category == "safety"

    def test_all_builtins_load(self):
        for name in BUILTIN_SUITES:
            suite = PromptSuite.builtin(name)
            assert len(suite) > 0

    def test_unknown_builtin_raises(self):
        with pytest.raises(PromptSuiteError, match="Unknown builtin suite"):
            PromptSuite.builtin("nonexistent")

    def test_load_from_yaml(self, tmp_path):
        data = {
            "name": "test-suite",
            "version": "1.0",
            "description": "Test suite",
            "category": "test",
            "prompts": [
                {"id": "t1", "text": "What is 1+1?", "expected_answer": "2"},
                {"id": "t2", "text": "What is 2+2?"},
            ],
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(data))

        suite = PromptSuite.from_yaml(path)
        assert suite.name == "test-suite"
        assert len(suite.prompts) == 2
        assert suite.prompts[0].expected_answer == "2"
        assert suite.prompts[1].expected_answer is None

    def test_load_via_name(self):
        suite = PromptSuite.load("math")
        assert suite.category == "math"

    def test_load_via_path(self, tmp_path):
        data = {
            "name": "custom",
            "category": "custom",
            "prompts": [{"id": "c1", "text": "Hello?"}],
        }
        path = tmp_path / "custom.yaml"
        path.write_text(yaml.dump(data))

        suite = PromptSuite.load(str(path))
        assert suite.name == "custom"

    def test_empty_prompts_raises(self, tmp_path):
        data = {"name": "empty", "category": "test", "prompts": []}
        path = tmp_path / "empty.yaml"
        path.write_text(yaml.dump(data))

        with pytest.raises(PromptSuiteError, match="no prompts"):
            PromptSuite.from_yaml(path)

    def test_duplicate_ids_raises(self, tmp_path):
        data = {
            "name": "dup",
            "category": "test",
            "prompts": [
                {"id": "same", "text": "A"},
                {"id": "same", "text": "B"},
            ],
        }
        path = tmp_path / "dup.yaml"
        path.write_text(yaml.dump(data))

        with pytest.raises(PromptSuiteError, match="Duplicate"):
            PromptSuite.from_yaml(path)

    def test_prompt_tags_are_tuple(self):
        suite = PromptSuite.builtin("math")
        for prompt in suite.prompts:
            assert isinstance(prompt.tags, tuple)
