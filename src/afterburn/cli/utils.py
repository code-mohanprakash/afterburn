"""CLI helpers for progress bars and formatting."""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


def create_progress() -> Progress:
    """Create a Rich progress bar for CLI output."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def print_header() -> None:
    """Print the Afterburn CLI header."""
    console.print()
    console.print("[bold orange3]Afterburn[/] — Post-training diagnostics for LLMs")
    console.print()


def print_risk_score(score: float, risk_level: str) -> None:
    """Print a formatted risk score."""
    color_map = {
        "low": "green",
        "moderate": "yellow",
        "high": "red",
        "critical": "bold red",
    }
    color = color_map.get(risk_level, "white")
    console.print(
        f"  Reward Hacking Risk: [{color}]{risk_level.upper()}[/] "
        f"([bold]{score:.0f}/100[/])"
    )


def print_error(message: str) -> None:
    """Print a formatted error message."""
    console.print(f"[bold red]Error:[/] {message}")


def print_success(message: str) -> None:
    """Print a formatted success message."""
    console.print(f"[bold green]✓[/] {message}")
