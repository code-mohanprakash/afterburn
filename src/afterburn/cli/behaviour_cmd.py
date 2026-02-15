"""CLI command: afterburn behaviour — behavioural analysis only."""

from __future__ import annotations

import click

from afterburn.cli.utils import console, create_progress, print_error, print_header
from afterburn.exceptions import AfterburnError


@click.command("behaviour")
@click.option("--base", required=True, help="Base model (HuggingFace ID or local path)")
@click.option("--trained", required=True, help="Post-trained model (HuggingFace ID or local path)")
@click.option(
    "--suites",
    multiple=True,
    default=None,
    help="Prompt suites (math, code, reasoning, safety)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "mps", "cpu"]),
    default=None,
    help="Compute device",
)
@click.option("-o", "--output", default=None, help="Output file path")
def behaviour(base, trained, suites, device, output):
    """Run behavioural shift analysis on a model pair."""
    from afterburn.behaviour.analyser import BehaviourAnalyser
    from afterburn.device import auto_detect_device
    from afterburn.types import ModelPair

    print_header()

    try:
        device_config = auto_detect_device(force_device=device)
        model_pair = ModelPair(base_model=base, trained_model=trained)

        analyser = BehaviourAnalyser(
            model_pair,
            device_config,
            suites=list(suites) if suites else None,
        )

        with create_progress() as progress:
            task = progress.add_task("Running behaviour analysis...", total=100)

            def on_progress(step: str, current: int, total: int):
                pct = int(current / max(total, 1) * 100)
                progress.update(task, completed=pct, description=step)

            analyser._progress = on_progress
            result = analyser.run()
            progress.update(task, completed=100, description="Complete")

        # Display results
        console.print()
        console.print("[bold]Behaviour Analysis Results[/]")
        console.print()
        console.print(f"  {result.summary}")
        console.print()

        la = result.length_analysis
        console.print(f"  [bold]Output Length:[/]")
        console.print(f"    Base mean:    {la.base_mean:.1f} tokens")
        console.print(f"    Trained mean: {la.trained_mean:.1f} tokens")
        console.print(f"    Cohen's d:    {la.cohens_d:.2f}")
        console.print(f"    p-value:      {la.p_value:.4f}")
        console.print(f"    Significant:  {'Yes' if la.is_significant else 'No'}")
        console.print()

        sa = result.strategy_analysis
        if sa.dominant_shift:
            console.print(f"  [bold]Strategy Shift:[/] {sa.dominant_shift}")
        console.print(
            f"  Strategy entropy: {sa.base_entropy:.2f} → {sa.trained_entropy:.2f}"
        )

        if output:
            from pathlib import Path
            from afterburn.types import DiagnosticReport

            report = DiagnosticReport(model_pair=model_pair, behaviour=result)
            out_path = report.save(output)
            console.print(f"\n[green]Report saved to {out_path}[/]")

        console.print()

    except AfterburnError as e:
        print_error(str(e))
        raise SystemExit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        raise SystemExit(130)
