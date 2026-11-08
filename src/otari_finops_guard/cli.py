"""Interactive CLI for Otari FinOps Guard & Adaptive Model Router."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import typer
import uvicorn
from rich.console import Console

from otari_finops_guard.client import OtariFinOpsClient
from otari_finops_guard.dashboard import create_dashboard_app, render_terminal_dashboard
from otari_finops_guard.mock_gateway import create_mock_otari_app
from otari_finops_guard.swarm import run_deterministic_swarm_demo

app = typer.Typer(
    name="otari-finops-guard",
    help="Per-Task & Per-Subagent Virtual Key Provisioning & Adaptive Open-Weight Model Router for Mozilla Otari.",
    add_completion=False,
)


@app.command("demo")
def cmd_demo(
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="OTARI_BASE_URL",
        help="Live Mozilla Otari gateway URL (uses embedded deterministic Otari gateway if omitted).",
    ),
    api_key: str = typer.Option(
        "otari_sk_service_demo_key",
        "--api-key",
        envvar="OTARI_API_KEY",
        help="Otari service or virtual key.",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("artifacts"),
        "--output-dir",
        help="Directory for checkpoint and telemetry artifacts.",
    ),
) -> None:
    """Run the 7-step multi-agent coding swarm demo and render the terminal telemetry dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "task-otari-issue-2992.checkpoint.json"
    report = asyncio.run(
        run_deterministic_swarm_demo(
            base_url=base_url,
            api_key=api_key,
            checkpoint_path=checkpoint_path,
        )
    )
    render_terminal_dashboard(report)


@app.command("serve-dashboard")
def cmd_serve_dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8085, "--port", help="Bind port."),
) -> None:
    """Start the self-contained Web Telemetry Dashboard server."""
    report = asyncio.run(run_deterministic_swarm_demo())
    web_app = create_dashboard_app(initial_report=report)
    uvicorn.run(web_app, host=host, port=port)


@app.command("serve-mock-gateway")
def cmd_serve_mock_gateway(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
) -> None:
    """Start the standalone deterministic Mock Mozilla Otari Gateway server."""
    gw_app = create_mock_otari_app()
    uvicorn.run(gw_app, host=host, port=port)


@app.command("inspect-budget")
def cmd_inspect_budget(
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="OTARI_BASE_URL",
        help="Live Mozilla Otari gateway URL (uses embedded deterministic gateway if omitted).",
    ),
    api_key: str = typer.Option(
        "otari_sk_service_demo_key",
        "--api-key",
        envvar="OTARI_API_KEY",
        help="Otari API key.",
    ),
) -> None:
    """Inspect GET /api/v1/key/budget and GET /api/v1/end-user-budgets."""
    console = Console()

    async def _inspect() -> None:
        if base_url:
            async with OtariFinOpsClient(base_url=base_url, api_key=api_key) as client:
                kb = await client.get_key_budget()
                eub = await client.list_end_user_budgets()
        else:
            gw_app = create_mock_otari_app()
            async with OtariFinOpsClient(
                base_url="http://otari.local",
                api_key=api_key,
                transport=httpx.ASGITransport(app=gw_app),
            ) as client:
                kb = await client.get_key_budget()
                eub = await client.list_end_user_budgets()
        console.print_json(
            data={
                "key_budget": kb.model_dump(mode="json"),
                "assignable_end_user_budgets": [b.model_dump(mode="json") for b in eub],
            }
        )

    asyncio.run(_inspect())


if __name__ == "__main__":
    app()
