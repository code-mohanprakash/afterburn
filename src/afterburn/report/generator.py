"""Report format router."""

from __future__ import annotations

import logging
from pathlib import Path

from afterburn.exceptions import ReportGenerationError
from afterburn.report.html_report import HTMLReport
from afterburn.report.json_report import JSONReport
from afterburn.report.markdown_report import MarkdownReport
from afterburn.report.pdf_report import PDFReport
from afterburn.types import DiagnosticReport, ReportFormat

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Routes report generation to the appropriate format handler."""

    def __init__(self, report: DiagnosticReport):
        self.report = report

    def generate(self, fmt: ReportFormat, output_path: Path) -> Path:
        """Generate report in the specified format."""
        match fmt:
            case ReportFormat.HTML:
                return HTMLReport(self.report).generate(output_path)
            case ReportFormat.JSON:
                return JSONReport(self.report).generate(output_path)
            case ReportFormat.MARKDOWN:
                return MarkdownReport(self.report).generate(output_path)
            case ReportFormat.PDF:
                return PDFReport(self.report).generate(output_path)
            case _:
                raise ReportGenerationError(f"Unsupported report format: {fmt}")
