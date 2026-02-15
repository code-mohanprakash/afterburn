"""Prompt suite loading and management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from afterburn.exceptions import PromptSuiteError
from afterburn.prompts.schema import validate_suite_data
from afterburn.types import Prompt

logger = logging.getLogger(__name__)

BUILTIN_DIR = Path(__file__).parent / "builtin"
BUILTIN_SUITES = ["math", "code", "reasoning", "safety"]


@dataclass
class PromptSuite:
    """A collection of prompts for model evaluation."""

    name: str
    version: str
    description: str
    category: str
    prompts: list[Prompt]

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PromptSuite:
        """Load a prompt suite from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise PromptSuiteError(f"Prompt suite file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise PromptSuiteError(f"Invalid YAML in {path}: {e}") from e

        validate_suite_data(data, source=str(path))
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: Path | str) -> PromptSuite:
        """Load a prompt suite from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise PromptSuiteError(f"Prompt suite file not found: {path}")

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise PromptSuiteError(f"Invalid JSON in {path}: {e}") from e

        validate_suite_data(data, source=str(path))
        return cls._from_dict(data)

    @classmethod
    def builtin(cls, name: str) -> PromptSuite:
        """Load a built-in prompt suite by name.

        Available: math, code, reasoning, safety.
        """
        if name not in BUILTIN_SUITES:
            available = ", ".join(BUILTIN_SUITES)
            raise PromptSuiteError(
                f"Unknown builtin suite: '{name}'. Available: {available}"
            )

        path = BUILTIN_DIR / f"{name}.yaml"
        if not path.exists():
            raise PromptSuiteError(f"Builtin suite file missing: {path}")

        return cls.from_yaml(path)

    @classmethod
    def load(cls, name_or_path: str) -> PromptSuite:
        """Load a suite by name (builtin) or path (custom).

        If the name matches a builtin suite, loads that.
        Otherwise, treats it as a file path.
        """
        if name_or_path in BUILTIN_SUITES:
            return cls.builtin(name_or_path)

        path = Path(name_or_path)
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif path.suffix == ".json":
            return cls.from_json(path)
        else:
            raise PromptSuiteError(
                f"Cannot determine format of '{name_or_path}'. "
                "Use .yaml/.yml or .json extension."
            )

    @classmethod
    def _from_dict(cls, data: dict) -> PromptSuite:
        """Convert validated dict to PromptSuite."""
        category = data["category"]
        prompts = [
            Prompt(
                id=p["id"],
                text=p["text"],
                category=p.get("category", category),
                expected_answer=p.get("expected_answer"),
                difficulty=p.get("difficulty"),
                tags=tuple(p.get("tags", [])),
            )
            for p in data["prompts"]
        ]

        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            category=category,
            prompts=prompts,
        )
