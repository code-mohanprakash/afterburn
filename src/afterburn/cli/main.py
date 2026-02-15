"""Afterburn CLI entry point."""

from __future__ import annotations

import logging

import click

from afterburn.version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="afterburn")
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging"
)
def cli(verbose: bool) -> None:
    """Afterburn: Post-training diagnostics for LLMs.

    Analyze what post-training actually did to your model.

    Quick start:

        afterburn diagnose --base meta-llama/Llama-3.1-8B --trained my-org/Llama-RLVR -o report.html
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# Import and register subcommands
from afterburn.cli.diagnose import diagnose
from afterburn.cli.weight_diff_cmd import weight_diff
from afterburn.cli.behaviour_cmd import behaviour
from afterburn.cli.hack_check_cmd import hack_check

cli.add_command(diagnose)
cli.add_command(weight_diff)
cli.add_command(behaviour)
cli.add_command(hack_check)
