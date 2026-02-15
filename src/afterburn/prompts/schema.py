"""Prompt suite YAML/JSON schema validation."""

from __future__ import annotations

from typing import Any

from afterburn.exceptions import PromptSuiteError

REQUIRED_TOP_LEVEL = {"name", "category", "prompts"}
REQUIRED_PROMPT_FIELDS = {"id", "text"}


def validate_suite_data(data: Any, source: str = "<unknown>") -> None:
    """Validate raw suite data (parsed from YAML/JSON).

    Raises PromptSuiteError if validation fails.
    """
    if not isinstance(data, dict):
        raise PromptSuiteError(
            f"Prompt suite {source} must be a YAML/JSON mapping, got {type(data).__name__}"
        )

    missing = REQUIRED_TOP_LEVEL - set(data.keys())
    if missing:
        raise PromptSuiteError(
            f"Prompt suite {source} missing required fields: {missing}"
        )

    prompts = data["prompts"]
    if not isinstance(prompts, list):
        raise PromptSuiteError(
            f"'prompts' in {source} must be a list, got {type(prompts).__name__}"
        )

    if len(prompts) == 0:
        raise PromptSuiteError(f"Prompt suite {source} has no prompts")

    seen_ids: set[str] = set()
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise PromptSuiteError(
                f"Prompt #{i} in {source} must be a mapping, got {type(prompt).__name__}"
            )

        missing_fields = REQUIRED_PROMPT_FIELDS - set(prompt.keys())
        if missing_fields:
            raise PromptSuiteError(
                f"Prompt #{i} in {source} missing required fields: {missing_fields}"
            )

        pid = prompt["id"]
        if pid in seen_ids:
            raise PromptSuiteError(
                f"Duplicate prompt ID '{pid}' in {source}"
            )
        seen_ids.add(pid)

        if not isinstance(prompt["text"], str) or not prompt["text"].strip():
            raise PromptSuiteError(
                f"Prompt '{pid}' in {source} has empty or invalid text"
            )
