"""1-Command Zero-Config Live Demo for Otari FinOps Guard & Adaptive Model Router."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from rich.console import Console

from otari_finops_guard.dashboard import render_terminal_dashboard
from otari_finops_guard.swarm import run_deterministic_swarm_demo


async def _run_demo_async(
    *,
    base_url: str | None = None,
    api_key: str = "otari_sk_service_demo_key",
    output_dir: Path | None = None,
) -> int:
    console = Console(width=116)
    target_dir = output_dir or Path("artifacts")
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / "task-otari-issue-2992.checkpoint.json"
    report_path = target_dir / "swarm-telemetry-report.json"

    report = await run_deterministic_swarm_demo(
        base_url=base_url,
        api_key=api_key,
        checkpoint_path=checkpoint_path,
    )
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    render_terminal_dashboard(report, console=console)
    console.print(
        f"\n[bold green]✔ Saved deterministic checkpoint:[/bold green] {checkpoint_path}\n"
        f"[bold green]✔ Saved full telemetry report:[/bold green]    {report_path}\n"
        "[dim]Launch interactive Web Telemetry Console with:[/dim] "
        "[bold cyan]python3 -m otari_finops_guard.cli serve-dashboard --port 8085[/bold cyan]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m otari_finops_guard.demo",
        description="Run the Otari FinOps Guard & Adaptive Model Router multi-agent swarm demo.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OTARI_BASE_URL"),
        help="Optional live Mozilla Otari gateway URL (defaults to embedded deterministic Otari ASGI gateway).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OTARI_API_KEY", "otari_sk_service_demo_key"),
        help="Otari service/virtual API key.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write checkpoint and telemetry JSON artifacts.",
    )
    args = parser.parse_args(argv)
    return asyncio.run(
        _run_demo_async(
            base_url=args.base_url,
            api_key=args.api_key,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
