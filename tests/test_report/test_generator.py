"""Tests for ReportGenerator router."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from afterburn.exceptions import ReportGenerationError
from afterburn.report.generator import ReportGenerator
from afterburn.types import DiagnosticReport, ModelPair, ReportFormat, TrainingMethod


@pytest.fixture
def sample_report():
    """Create a sample DiagnosticReport for testing."""
    model_pair = ModelPair(
        base_model="test/base",
        trained_model="test/trained",
        method=TrainingMethod.SFT,
    )
    return DiagnosticReport(
        model_pair=model_pair,
        summary="Test report for generator",
    )


def test_generator_routes_to_html_report(sample_report, tmp_path):
    """Test that ReportFormat.HTML routes to HTMLReport."""
    output_path = tmp_path / "report.html"

    with patch("afterburn.report.generator.HTMLReport") as mock_html_class:
        mock_html_instance = MagicMock()
        mock_html_instance.generate.return_value = output_path
        mock_html_class.return_value = mock_html_instance

        generator = ReportGenerator(sample_report)
        result = generator.generate(ReportFormat.HTML, output_path)

        # Verify HTMLReport was instantiated with the report
        mock_html_class.assert_called_once_with(sample_report)
        # Verify generate was called with the output path
        mock_html_instance.generate.assert_called_once_with(output_path)
        assert result == output_path


def test_generator_routes_to_json_report(sample_report, tmp_path):
    """Test that ReportFormat.JSON routes to JSONReport."""
    output_path = tmp_path / "report.json"

    with patch("afterburn.report.generator.JSONReport") as mock_json_class:
        mock_json_instance = MagicMock()
        mock_json_instance.generate.return_value = output_path
        mock_json_class.return_value = mock_json_instance

        generator = ReportGenerator(sample_report)
        result = generator.generate(ReportFormat.JSON, output_path)

        # Verify JSONReport was instantiated with the report
        mock_json_class.assert_called_once_with(sample_report)
        # Verify generate was called with the output path
        mock_json_instance.generate.assert_called_once_with(output_path)
        assert result == output_path


def test_generator_routes_to_markdown_report(sample_report, tmp_path):
    """Test that ReportFormat.MARKDOWN routes to MarkdownReport."""
    output_path = tmp_path / "report.md"

    with patch("afterburn.report.generator.MarkdownReport") as mock_md_class:
        mock_md_instance = MagicMock()
        mock_md_instance.generate.return_value = output_path
        mock_md_class.return_value = mock_md_instance

        generator = ReportGenerator(sample_report)
        result = generator.generate(ReportFormat.MARKDOWN, output_path)

        # Verify MarkdownReport was instantiated with the report
        mock_md_class.assert_called_once_with(sample_report)
        # Verify generate was called with the output path
        mock_md_instance.generate.assert_called_once_with(output_path)
        assert result == output_path


def test_generator_routes_to_pdf_report(sample_report, tmp_path):
    """Test that ReportFormat.PDF routes to PDFReport."""
    output_path = tmp_path / "report.pdf"

    with patch("afterburn.report.generator.PDFReport") as mock_pdf_class:
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.generate.return_value = output_path
        mock_pdf_class.return_value = mock_pdf_instance

        generator = ReportGenerator(sample_report)
        result = generator.generate(ReportFormat.PDF, output_path)

        # Verify PDFReport was instantiated with the report
        mock_pdf_class.assert_called_once_with(sample_report)
        # Verify generate was called with the output path
        mock_pdf_instance.generate.assert_called_once_with(output_path)
        assert result == output_path


def test_generator_invalid_format_raises_error(sample_report, tmp_path):
    """Test that an invalid format raises ReportGenerationError."""
    output_path = tmp_path / "report.txt"

    # Create an invalid enum-like object (simulating corrupted state)
    # We need to test the case statement's default branch
    # Since we can't create invalid enum values directly, we'll patch the match
    generator = ReportGenerator(sample_report)

    # This is a bit tricky - we need to trigger the default case
    # Let's create a mock format that won't match any case
    class FakeFormat:
        def __str__(self):
            return "FAKE"

    fake_format = FakeFormat()

    with pytest.raises(ReportGenerationError, match="Unsupported report format"):
        # This will raise because FakeFormat isn't a valid ReportFormat enum
        generator.generate(fake_format, output_path)


def test_generator_instantiation(sample_report):
    """Test that ReportGenerator can be instantiated with a report."""
    generator = ReportGenerator(sample_report)

    assert generator.report is sample_report


def test_generator_all_formats_have_handlers(sample_report, tmp_path):
    """Test that all ReportFormat enum values have handlers."""
    # This ensures we don't forget to add a handler when adding a new format
    for fmt in ReportFormat:
        output_path = tmp_path / f"report.{fmt.value}"

        # Mock the appropriate report class
        if fmt == ReportFormat.HTML:
            mock_class_path = "afterburn.report.generator.HTMLReport"
        elif fmt == ReportFormat.JSON:
            mock_class_path = "afterburn.report.generator.JSONReport"
        elif fmt == ReportFormat.MARKDOWN:
            mock_class_path = "afterburn.report.generator.MarkdownReport"
        elif fmt == ReportFormat.PDF:
            mock_class_path = "afterburn.report.generator.PDFReport"
        else:
            pytest.fail(f"Unknown format {fmt} - update test!")

        with patch(mock_class_path) as mock_class:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = output_path
            mock_class.return_value = mock_instance

            generator = ReportGenerator(sample_report)
            result = generator.generate(fmt, output_path)

            assert result == output_path
            mock_class.assert_called_once()
            mock_instance.generate.assert_called_once()
