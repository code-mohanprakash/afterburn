"""Afterburn Python API demo — reward hacking detection in 10 lines."""

from afterburn import Diagnoser

# Compare base vs fine-tuned model
diag = Diagnoser(
    base_model="distilbert/distilgpt2",
    trained_model="bkwalsh/distilgpt2-finetuned-wikitext2",
    method="sft",
    device="cpu",
)

# Run full diagnostics (weight diff + behaviour + reward hack)
print("Running diagnostics...")
report = diag.run()

# Print results
print(f"\nRisk Level: {report.reward_hack.risk_level.value.upper()}")
print(f"Risk Score: {report.hack_score:.0f}/100")
print()
print("Score Breakdown:")
print(f"  Length Bias:       {report.reward_hack.length_bias.score:.1f}/100")
print(f"  Format Gaming:    {report.reward_hack.format_gaming.score:.1f}/100")
print(f"  Strategy Collapse: {report.reward_hack.strategy_collapse.score:.1f}/100")
print(f"  Sycophancy:       {report.reward_hack.sycophancy.score:.1f}/100")

if report.reward_hack.flags:
    print("\nFlags:")
    for flag in report.reward_hack.flags:
        print(f"  - {flag}")

# Save interactive HTML report
report.save("demo/output/api-report.html")
print("\nReport saved to demo/output/api-report.html")
