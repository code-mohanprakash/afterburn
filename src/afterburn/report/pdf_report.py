"""PDF report generation via WeasyPrint (optional dependency)."""

from __future__ import annotations

import logging
from pathlib import Path

from afterburn.exceptions import ReportGenerationError
from afterburn.report.html_report import HTMLReport
from afterburn.types import DiagnosticReport

logger = logging.getLogger(__name__)


class PDFReport:
    """Generates PDF diagnostic reports.

    Requires the optional 'pdf' dependency: pip install afterburn[pdf]
    Uses WeasyPrint to convert HTML to PDF.
    """

    def __init__(self, report: DiagnosticReport):
        self.report = report

    def generate(self, output_path: Path) -> Path:
        """Generate PDF report file."""
        try:
            import weasyprint
        except ImportError:
            raise ReportGenerationError(
                "PDF generation requires WeasyPrint. "
                "Install it with: pip install afterburn[pdf]"
            )

        try:
            # First generate HTML
            html_report = HTMLReport(self.report)
            # Generate to a temp HTML string
            from jinja2 import Environment, FileSystemLoader
            from afterburn.report.html_report import TEMPLATE_DIR
            from afterburn.version import __version__
            from datetime import datetime

            env = Environment(
                loader=FileSystemLoader(str(TEMPLATE_DIR)),
                autoescape=True,
            )
            template = env.get_template("report.html.j2")
            css_path = TEMPLATE_DIR / "styles.css"
            styles = css_path.read_text() if css_path.exists() else ""
            context = html_report._build_context(styles)
            html_content = template.render(**context)

            # Convert to PDF
            output_path.parent.mkdir(parents=True, exist_ok=True)
            doc = weasyprint.HTML(string=html_content)
            doc.write_pdf(str(output_path))

            logger.info("PDF report saved to %s", output_path)
            return output_path

        except Exception as e:
            if "weasyprint" in str(type(e).__module__).lower():
                raise ReportGenerationError(
                    f"WeasyPrint failed to generate PDF: {e}. "
                    f"Consider using HTML format instead."
                ) from e
            raise ReportGenerationError(f"Failed to generate PDF report: {e}") from e
