"""CLI command: afterburn diagnose — full diagnostic analysis."""

from __future__ import annotations

from pathlib import Path

import click

from afterburn.cli.utils import (
    console,
    create_progress,
    print_error,
    print_header,
    print_risk_score,
    print_success,
)
from afterburn.exceptions import AfterburnError
from afterburn.types import ReportFormat


@click.command()
@click.option("--base", required=True, help="Base model (HuggingFace ID or local path)")
@click.option("--trained", required=True, help="Post-trained model (HuggingFace ID or local path)")
@click.option(
    "--method",
    type=click.Choice(["sft", "dpo", "rlhf", "rlvr", "grpo", "lora", "qlora", "unknown"]),
    default="unknown",
    help="Training method used",
)
@click.option("--output", "-o", default="report.html", help="Output file path")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["html", "markdown", "pdf", "json"]),
    default=None,
    help="Report format (auto-detected from extension if not set)",
)
@click.option(
    "--suites",
    multiple=True,
    default=None,
    help="Prompt suites to use (math, code, reasoning, safety, or path to custom YAML)",
)
@click.option(
    "--modules",
    multiple=True,
    default=None,
    help="Analysis modules to run (weight_diff, behaviour, reward_hack)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "mps", "cpu"]),
    default=None,
    help="Compute device (auto-detected if not set)",
)
@click.option("--config", type=click.Path(exists=True), default=None, help="Config file path")
@click.option(
    "--fast",
    is_flag=True,
    default=False,
    help="Fast mode: 2 suites + 128 tokens (~16x faster)",
)
def diagnose(
    base: str,
    trained: str,
    method: str,
    output: str,
    fmt: str,
    suites: tuple[str, ...],
    modules: tuple[str, ...],
    device: str,
    config: str,
    fast: bool,
) -> None:
    """Run full diagnostic analysis on a base/trained model pair."""
    from afterburn.diagnoser import Diagnoser

    print_header()

    console.print(f"  Base:    [cyan]{base}[/]")
    console.print(f"  Trained: [cyan]{trained}[/]")
    console.print(f"  Method:  [cyan]{method.upper()}[/]")
    console.print()

    try:
        diag = Diagnoser(
            base_model=base,
            trained_model=trained,
            method=method,
            suites=list(suites) if suites else None,
            modules=list(modules) if modules else None,
            device=device,
            config_path=config,
            fast=fast,
        )

        with create_progress() as progress:
            task = progress.add_task("Running diagnostics...", total=100)

            def on_progress(step: str, current: int, total: int) -> None:
                pct = int(current / max(total, 1) * 100)
                progress.update(task, completed=pct, description=step)

            report = diag.run(progress=True, progress_callback=on_progress)
            progress.update(task, completed=100, description="Complete")

        # Save report
        output_path = Path(output)
        report_format = ReportFormat(fmt) if fmt else None  # Auto-detect

        report.save(output_path, fmt=report_format)

        # Print summary
        console.print()
        print_success(f"Report saved to [bold]{output_path}[/]")
        console.print()

        if report.reward_hack:
            print_risk_score(report.hack_score, report.reward_hack.risk_level.value)

        if report.weight_diff:
            top = report.weight_diff.top_changed_layers
            if top:
                console.print(
                    f"  Most changed layer: [bold]{top[0].layer_name}[/] "
                    f"(relative change: {top[0].relative_change:.6f})"
                )

        console.print()
        summary_text = (
            f"  [dim]{report.summary[:200]}...[/]"
            if len(report.summary) > 200
            else f"  [dim]{report.summary}[/]"
        )
        console.print(summary_text)
        console.print()

    except AfterburnError as e:
        print_error(str(e))
        raise SystemExit(1) from e
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        raise SystemExit(130) from None
