"""Tests for CLI commands."""

from click.testing import CliRunner

from afterburn.cli.main import cli


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "afterburn" in result.output.lower()

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "diagnostics" in result.output.lower()

    def test_diagnose_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["diagnose", "--help"])
        assert result.exit_code == 0
        assert "--base" in result.output
        assert "--trained" in result.output

    def test_weight_diff_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["weight-diff", "--help"])
        assert result.exit_code == 0
        assert "--base" in result.output

    def test_behaviour_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["behaviour", "--help"])
        assert result.exit_code == 0

    def test_hack_check_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hack-check", "--help"])
        assert result.exit_code == 0

    def test_diagnose_missing_args(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["diagnose"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower()
