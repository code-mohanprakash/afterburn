"""Tests for CLI entry point (afterburn --version, --help, subcommands)."""

from __future__ import annotations

from click.testing import CliRunner

from afterburn.cli.main import cli
from afterburn.version import __version__


def test_cli_version():
    """Test that --version outputs version string."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output
    assert "afterburn" in result.output.lower()


def test_cli_help():
    """Test that --help shows help text."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Afterburn" in result.output
    assert "Post-training diagnostics" in result.output or "post-training" in result.output.lower()
    assert "Commands:" in result.output or "Usage:" in result.output


def test_cli_subcommands_registered():
    """Test that all expected subcommands are registered."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    # All subcommands should appear in help
    assert "diagnose" in result.output
    assert "weight-diff" in result.output
    assert "behaviour" in result.output
    assert "hack-check" in result.output


def test_cli_verbose_flag():
    """Test that --verbose flag is accepted."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--verbose", "--help"])

    # Should still show help with verbose flag
    assert result.exit_code == 0
    assert "Afterburn" in result.output


def test_cli_short_verbose_flag():
    """Test that -v short flag is accepted."""
    runner = CliRunner()
    result = runner.invoke(cli, ["-v", "--help"])

    # Should still show help with -v flag
    assert result.exit_code == 0
    assert "Afterburn" in result.output
