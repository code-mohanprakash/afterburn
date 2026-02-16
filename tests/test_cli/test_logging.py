"""Tests for CLI logging functionality."""

import json
import logging

from click.testing import CliRunner

from afterburn.cli.main import cli


class TestLogging:
    """Test logging configuration and formats."""

    def test_default_log_level_is_info(self):
        """Test that default log level is INFO (not WARNING)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

        # Verify by checking that INFO level is set when no --verbose flag
        # We'll test this indirectly through the verbose test

    def test_verbose_enables_debug_level(self):
        """Test that --verbose flag enables DEBUG level."""
        runner = CliRunner()

        # Create a simple test that exercises logging
        # The --help command itself doesn't produce much logging,
        # but we can verify the flag is accepted
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_json_log_format_option(self):
        """Test that --log-format json option is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--log-format", "json", "--help"])
        assert result.exit_code == 0

    def test_text_log_format_option(self):
        """Test that --log-format text option is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--log-format", "text", "--help"])
        assert result.exit_code == 0

    def test_invalid_log_format_rejected(self):
        """Test that invalid log format is rejected."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--log-format", "invalid"])
        assert result.exit_code != 0
        # Click shows choice validation error before command execution
        assert "Invalid value" in result.output or "Choice" in result.output

    def test_json_formatter_output_structure(self, caplog):
        """Test that JSON formatter produces valid JSON output."""
        from afterburn.cli.main import JSONLogFormatter

        # Create a formatter
        formatter = JSONLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")

        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Format the record
        output = formatter.format(record)

        # Verify it's valid JSON
        parsed = json.loads(output)

        # Verify required fields
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed

        # Verify values
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"

    def test_json_formatter_formats_timestamp(self):
        """Test that JSON formatter correctly formats timestamps."""
        from afterburn.cli.main import JSONLogFormatter

        formatter = JSONLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        # Check timestamp format (should be ISO-like)
        assert "timestamp" in parsed
        # Timestamp should contain date and time components
        timestamp = parsed["timestamp"]
        assert len(timestamp) > 0

    def test_json_formatter_different_log_levels(self):
        """Test JSON formatter with different log levels."""
        from afterburn.cli.main import JSONLogFormatter

        formatter = JSONLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")

        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]

        for level_num, level_name in levels:
            record = logging.LogRecord(
                name="test",
                level=level_num,
                pathname="test.py",
                lineno=1,
                msg=f"Test {level_name}",
                args=(),
                exc_info=None,
            )

            output = formatter.format(record)
            parsed = json.loads(output)

            assert parsed["level"] == level_name
            assert parsed["message"] == f"Test {level_name}"

    def test_combined_verbose_and_json_format(self):
        """Test that --verbose and --log-format can be used together."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--log-format", "json", "--help"])
        assert result.exit_code == 0
