"""Afterburn CLI entry point."""

from __future__ import annotations

import json
import logging

import click

from afterburn.version import __version__


class JSONLogFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


@click.group()
@click.version_option(version=__version__, prog_name="afterburn")
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging"
)
@click.option(
    "--log-format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Log output format (text or json)",
)
def cli(verbose: bool, log_format: str) -> None:
    """Afterburn: Post-training diagnostics for LLMs.

    Analyze what post-training actually did to your model.

    Quick start:

        afterburn diagnose --base meta-llama/Llama-3.1-8B --trained my-org/Llama-RLVR -o report.html
    """
    level = logging.DEBUG if verbose else logging.INFO

    if log_format == "json":
        formatter = JSONLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logging.basicConfig(level=level, handlers=[handler])
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )


# Import and register subcommands
from afterburn.cli.behaviour_cmd import behaviour  # noqa: E402
from afterburn.cli.diagnose import diagnose  # noqa: E402
from afterburn.cli.hack_check_cmd import hack_check  # noqa: E402
from afterburn.cli.weight_diff_cmd import weight_diff  # noqa: E402

cli.add_command(diagnose)
cli.add_command(weight_diff)
cli.add_command(behaviour)
cli.add_command(hack_check)
