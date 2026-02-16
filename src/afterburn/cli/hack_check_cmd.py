"""CLI command: afterburn hack-check — reward hacking detection only."""

from __future__ import annotations

import click

from afterburn.cli.utils import (
    console,
    create_progress,
    print_error,
    print_header,
    print_risk_score,
)
from afterburn.exceptions import AfterburnError


@click.command("hack-check")
@click.option("--base", required=True, help="Base model (HuggingFace ID or local path)")
@click.option("--trained", required=True, help="Post-trained model (HuggingFace ID or local path)")
@click.option(
    "--method",
    type=click.Choice(["sft", "dpo", "rlhf", "rlvr", "grpo", "unknown"]),
    default="unknown",
    help="Training method",
)
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
def hack_check(
    base: str,
    trained: str,
    method: str,
    suites: tuple[str, ...],
    device: str,
    output: str,
) -> None:
    """Check for reward hacking patterns in a trained model."""
    from afterburn.diagnoser import Diagnoser

    print_header()

    try:
        diag = Diagnoser(
            base_model=base,
            trained_model=trained,
            method=method,
            suites=list(suites) if suites else None,
            modules=["behaviour", "reward_hack"],
            device=device,
        )

        with create_progress() as progress:
            task = progress.add_task("Checking for reward hacking...", total=100)

            def on_progress(step: str, current: int, total: int) -> None:
                pct = int(current / max(total, 1) * 100)
                progress.update(task, completed=pct, description=step)

            report = diag.run(progress=True, progress_callback=on_progress)
            progress.update(task, completed=100, description="Complete")

        # Display results
        console.print()
        if report.reward_hack:
            rh = report.reward_hack
            print_risk_score(rh.composite_score, rh.risk_level.value)
            console.print()

            # Sub-scores
            console.print("  [bold]Score Breakdown:[/]")
            lb_flag = "⚠" if rh.length_bias.is_flagged else "✓"
            console.print(f"    Length Bias:       {rh.length_bias.score:5.1f}/100  {lb_flag}")
            fg_flag = "⚠" if rh.format_gaming.is_flagged else "✓"
            console.print(f"    Format Gaming:    {rh.format_gaming.score:5.1f}/100  {fg_flag}")
            sc_flag = "⚠" if rh.strategy_collapse.is_flagged else "✓"
            console.print(
                f"    Strategy Collapse: {rh.strategy_collapse.score:5.1f}/100  {sc_flag}"
            )
            syc_flag = "⚠" if rh.sycophancy.is_flagged else "✓"
            console.print(f"    Sycophancy:       {rh.sycophancy.score:5.1f}/100  {syc_flag}")
            console.print()

            if rh.flags:
                console.print("  [bold yellow]Flags:[/]")
                for flag in rh.flags:
                    console.print(f"    ⚠ {flag}")
                console.print()

        if output:
            out_path = report.save(output)
            console.print(f"[green]Report saved to {out_path}[/]")

        console.print()

    except AfterburnError as e:
        print_error(str(e))
        raise SystemExit(1) from e
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        raise SystemExit(130) from None
