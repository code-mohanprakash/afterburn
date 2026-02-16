"""CLI command: afterburn weight-diff — weight-only analysis."""

from __future__ import annotations

import click

from afterburn.cli.utils import console, create_progress, print_error, print_header
from afterburn.exceptions import AfterburnError


@click.command("weight-diff")
@click.option("--base", required=True, help="Base model (HuggingFace ID or local path)")
@click.option("--trained", required=True, help="Post-trained model (HuggingFace ID or local path)")
@click.option(
    "--device",
    type=click.Choice(["cuda", "mps", "cpu"]),
    default=None,
    help="Compute device",
)
@click.option("--top-n", default=5, help="Number of top changed layers to show")
@click.option("-o", "--output", default=None, help="Output file path (json or html)")
def weight_diff(base: str, trained: str, device: str, top_n: int, output: str) -> None:
    """Analyze weight differences between base and trained models."""
    from afterburn.device import auto_detect_device
    from afterburn.types import ModelPair
    from afterburn.weight_diff.engine import WeightDiffEngine

    print_header()

    try:
        device_config = auto_detect_device(force_device=device)
        model_pair = ModelPair(base_model=base, trained_model=trained)

        with create_progress() as progress:
            task = progress.add_task("Analyzing weights...", total=100)

            def on_progress(step: str, current: int, total: int) -> None:
                pct = int(current / max(total, 1) * 100)
                progress.update(task, completed=pct, description=f"Layer: {step}")

            engine = WeightDiffEngine(model_pair, device_config, progress_callback=on_progress)
            result = engine.run()
            progress.update(task, completed=100, description="Complete")

        # Display results
        console.print()
        console.print("[bold]Weight Diff Results[/]")
        console.print(f"  Total params:   {result.total_param_count:,}")
        console.print(f"  Changed params: {result.changed_param_count:,}")
        console.print(f"  Concentration:  {result.change_concentration:.1%}")
        console.print()

        console.print(f"[bold]Top {top_n} Changed Layers:[/]")
        for i, layer in enumerate(result.top_changed_layers[:top_n], 1):
            console.print(
                f"  {i}. [cyan]{layer.layer_name}[/] "
                f"— L2: {layer.l2_norm:.4f}, "
                f"Cosine: {layer.cosine_similarity:.6f}, "
                f"Rel: {layer.relative_change:.6f}"
            )

        sig_norms = [s for s in result.layernorm_shifts if s.is_significant]
        if sig_norms:
            console.print()
            console.print(f"[bold yellow]⚠ {len(sig_norms)} significant LayerNorm shift(s)[/]")

        # Save to file if requested
        if output:
            from afterburn.types import DiagnosticReport

            report = DiagnosticReport(model_pair=model_pair, weight_diff=result)
            out_path = report.save(output)
            console.print(f"\n[green]Report saved to {out_path}[/]")

        console.print()

    except AfterburnError as e:
        print_error(str(e))
        raise SystemExit(1) from e
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        raise SystemExit(130) from None
