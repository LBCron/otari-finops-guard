"""Deterministic Mock Mozilla Otari Gateway (FastAPI ASGI Application).

Implements the exact Mozilla Otari HTTP contract (/api/v1/chat/completions,
/api/v1/messages, /api/v1/key/budget, /api/v1/key/usage/summary,
/api/v1/end-user-budgets, /api/v1/end-users, /api/v1/routing/recommend,
/api/v1/search, /api/v1/fetch, /api/v1/code/execute, /api/v1/mcp/servers,
and /api/v1/mcp/execute) with upfront worst-case budget reservation holds,
service-key per-user budgets, guardrails, idempotency replay, and upstream
issue reproduction toggles (#2992, #2989, #2981).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from otari_finops_guard.models import (
    EndUserBudgetPublic,
    EndUserPublic,
    KeyBudgetLimit,
    KeyBudgetState,
    OtariToolCall,
    UsageSummaryBucket,
    UsageSummaryResponse,
)
from otari_finops_guard.router import OPEN_WEIGHT_CATALOG

MODEL_PRICING_PER_1K: dict[str, tuple[Decimal, Decimal]] = {
    candidate.target: (candidate.input_cost_per_1k, candidate.output_cost_per_1k)
    for pair in OPEN_WEIGHT_CATALOG.values()
    for candidate in pair
}
MODEL_PRICING_PER_1K.update(
    {
        "deepseek-r1": (Decimal("0.00055"), Decimal("0.00219")),
        "Qwen/Qwen2.5-Coder-32B-Instruct": (Decimal("0.00020"), Decimal("0.00060")),
        "meta-llama/Llama-3.3-70B-Instruct": (Decimal("0.00018"), Decimal("0.00054")),
        "mistral-small-24b-instruct-2501": (Decimal("0.00007"), Decimal("0.00020")),
    }
)


@dataclass
class InternalEndUserRecord:
    """Internal User row mirroring Otari's users table (with eu_<hex> user_id)."""

    internal_user_id: str
    external_id: str
    budget_id: str
    max_budget: Decimal
    spend: Decimal = Decimal("0")
    reserved: Decimal = Decimal("0")
    token_limit: int | None = 50000
    current_tokens: int = 0
    reserved_tokens: int = 0
    request_limit: int | None = 100
    current_requests: int = 0
    reserved_requests: int = 0
    budget_duration: str = "1d"
    next_budget_reset_at: str = field(default_factory=lambda: (datetime.now(UTC) + timedelta(days=1)).isoformat())
    blocked: bool = False

    def to_public(self) -> EndUserPublic:
        remaining = max(Decimal("0"), self.max_budget - (self.spend + self.reserved))
        return EndUserPublic(
            external_id=self.external_id,
            budget_id=self.budget_id,
            max_budget=self.max_budget,
            spend=self.spend,
            reserved=self.reserved,
            remaining_budget=remaining,
            token_limit=self.token_limit,
            current_tokens=self.current_tokens,
            reserved_tokens=self.reserved_tokens,
            request_limit=self.request_limit,
            current_requests=self.current_requests,
            reserved_requests=self.reserved_requests,
            budget_duration=self.budget_duration,
            next_budget_reset_at=self.next_budget_reset_at,
            blocked=self.blocked,
        )


class MockOtariGatewayState:
    """In-memory state for the deterministic Mock Mozilla Otari Gateway."""

    def __init__(
        self,
        *,
        key_max_budget_usd: Decimal = Decimal("0.050000"),
        simulate_issue_2992_internal_id: bool = True,
        simulate_issue_2989_missing_header: bool = True,
    ) -> None:
        self.key_max_budget_usd = key_max_budget_usd
        self.key_spend_usd = Decimal("0")
        self.key_used_tokens = 0
        self.key_used_requests = 0
        self.simulate_issue_2992_internal_id = simulate_issue_2992_internal_id
        self.simulate_issue_2989_missing_header = simulate_issue_2989_missing_header
        self.simulate_first_candidate_provider_ceiling_exhausted: set[str] = set()
        self.simulate_transient_429_once_for_models: set[str] = set()

        self.budgets: dict[str, EndUserBudgetPublic] = {
            "subagent-architect-budget": EndUserBudgetPublic(
                budget_id="subagent-architect-budget",
                max_budget=Decimal("0.005000"),
                budget_duration="1d",
                token_limit=25000,
                request_limit=25,
                is_default=False,
            ),
            "subagent-researcher-budget": EndUserBudgetPublic(
                budget_id="subagent-researcher-budget",
                max_budget=Decimal("0.004000"),
                budget_duration="1d",
                token_limit=25000,
                request_limit=25,
                is_default=False,
            ),
            "subagent-coder-budget": EndUserBudgetPublic(
                budget_id="subagent-coder-budget",
                max_budget=Decimal("0.001750"),
                budget_duration="1d",
                token_limit=25000,
                request_limit=25,
                is_default=True,
            ),
            "subagent-verifier-budget": EndUserBudgetPublic(
                budget_id="subagent-verifier-budget",
                max_budget=Decimal("0.002000"),
                budget_duration="1d",
                token_limit=25000,
                request_limit=25,
                is_default=False,
            ),
        }
        self.end_users: dict[str, InternalEndUserRecord] = {}
        self.idempotency_store: dict[str, tuple[str, dict[str, Any], dict[str, str]]] = {}
        self.usage_buckets: dict[tuple[str, str, str], UsageSummaryBucket] = {}

    def get_or_create_end_user(self, external_id: str, requested_budget_id: str | None = None) -> InternalEndUserRecord:
        if external_id in self.end_users:
            return self.end_users[external_id]
        budget_id = requested_budget_id or "subagent-coder-budget"
        if budget_id not in self.budgets:
            raise HTTPException(
                status_code=403,
                detail=f"This key may not assign budget '{budget_id}' to its end users",
                headers={"Otari-Error-Code": "end_user_budget_not_allowed"},
            )
        budget = self.budgets[budget_id]
        digest = hashlib.md5(external_id.encode("utf-8")).hexdigest()
        record = InternalEndUserRecord(
            internal_user_id=f"eu_{digest}",
            external_id=external_id,
            budget_id=budget_id,
            max_budget=budget.max_budget or Decimal("0.002200"),
            token_limit=budget.token_limit,
            request_limit=budget.request_limit,
            budget_duration=budget.budget_duration or "1d",
        )
        self.end_users[external_id] = record
        return record


def create_mock_otari_app(state: MockOtariGatewayState | None = None) -> FastAPI:
    """Create the FastAPI application emulating a Mozilla Otari gateway."""
    gw = state or MockOtariGatewayState()
    app = FastAPI(title="Mozilla Otari Mock Gateway", version="1.0.0")
    app.state.gw = gw

    @app.get("/api/v1/key/budget", response_model=KeyBudgetState)
    async def get_key_budget() -> KeyBudgetState:
        remaining = max(Decimal("0"), gw.key_max_budget_usd - gw.key_spend_usd)
        resets_at = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        return KeyBudgetState(
            budgets_enforced=True,
            remaining_usd=remaining,
            remaining_tokens=max(0, 200000 - gw.key_used_tokens),
            remaining_requests=max(0, 500 - gw.key_used_requests),
            resets_at=resets_at,
            limits=[
                KeyBudgetLimit(
                    scope="user",
                    provider=None,
                    max_budget_usd=gw.key_max_budget_usd,
                    spend_usd=gw.key_spend_usd,
                    reserved_usd=Decimal("0"),
                    remaining_usd=remaining,
                    max_tokens=200000,
                    used_tokens=gw.key_used_tokens,
                    remaining_tokens=max(0, 200000 - gw.key_used_tokens),
                    max_requests=500,
                    used_requests=gw.key_used_requests,
                    remaining_requests=max(0, 500 - gw.key_used_requests),
                    budget_duration="1d",
                    resets_at=resets_at,
                )
            ],
        )

    @app.get("/api/v1/key/usage/summary", response_model=UsageSummaryResponse)
    async def get_key_usage_summary(
        group_by_tag: str | None = Query(default=None),
        dimensions: str | None = Query(default=None),
    ) -> UsageSummaryResponse:
        buckets = list(gw.usage_buckets.values())
        return UsageSummaryResponse(
            total_requests=sum(b.requests for b in buckets),
            total_input_tokens=sum(b.input_tokens for b in buckets),
            total_output_tokens=sum(b.output_tokens for b in buckets),
            total_tokens=sum(b.total_tokens for b in buckets),
            total_cost_usd=sum((b.cost_usd for b in buckets), Decimal("0")),
            buckets=buckets,
        )

    @app.get("/api/v1/end-user-budgets", response_model=list[EndUserBudgetPublic])
    async def list_end_user_budgets() -> list[EndUserBudgetPublic]:
        return list(gw.budgets.values())

    @app.get("/api/v1/end-users", response_model=EndUserPublic | list[EndUserPublic])
    async def get_end_users(external_id: str | None = Query(default=None)) -> Any:
        if external_id is not None:
            if external_id not in gw.end_users:
                raise HTTPException(status_code=404, detail=f"End user '{external_id}' not found")
            return gw.end_users[external_id].to_public()
        return [u.to_public() for u in gw.end_users.values()]

    @app.put("/api/v1/end-users", response_model=EndUserPublic)
    async def put_end_user(
        request: Request,
        external_id: str = Query(...),
    ) -> JSONResponse:
        body = await request.json()
        budget_id = body.get("budget_id", "subagent-coder-budget")
        created = external_id not in gw.end_users
        record = gw.get_or_create_end_user(external_id, budget_id)
        if not created and budget_id in gw.budgets:
            record.budget_id = budget_id
            record.max_budget = gw.budgets[budget_id].max_budget or record.max_budget
        return JSONResponse(
            status_code=201 if created else 200,
            content=json.loads(record.to_public().model_dump_json()),
        )

    @app.patch("/api/v1/end-users", response_model=EndUserPublic)
    async def patch_end_user(
        request: Request,
        external_id: str = Query(...),
    ) -> EndUserPublic:
        if external_id not in gw.end_users:
            raise HTTPException(status_code=404, detail=f"End user '{external_id}' not found")
        body = await request.json()
        record = gw.end_users[external_id]
        if "blocked" in body and body["blocked"] is not None:
            record.blocked = bool(body["blocked"])
        if "budget_id" in body and body["budget_id"] in gw.budgets:
            record.budget_id = body["budget_id"]
            record.max_budget = gw.budgets[body["budget_id"]].max_budget or record.max_budget
        return record.to_public()

    @app.post("/api/v1/routing/recommend")
    async def recommend_routing(request: Request) -> dict[str, Any]:
        body = await request.json()
        harness = body.get("harness", "pi")
        candidates: dict[str, str] = body.get("candidates", {})
        assignments: dict[str, str] = {}
        for target, desc in candidates.items():
            lower = desc.lower()
            if "architect" in lower or "r1" in target.lower():
                assignments.setdefault("architect", target)
            elif "research" in lower or "llama" in target.lower():
                assignments.setdefault("researcher", target)
            elif "coder" in lower or "qwen" in target.lower():
                assignments.setdefault("coder", target)
            elif "verif" in lower or "mistral" in target.lower():
                assignments.setdefault("verifier", target)
        return {
            "harness": harness,
            "recommendations": assignments,
            "fallback_order": list(candidates.keys()),
        }

    @app.post("/api/v1/search")
    async def web_search(request: Request) -> dict[str, Any]:
        body = await request.json()
        query = body.get("query", "")
        return {
            "query": query,
            "provider": "tavily",
            "cost_usd": "0.000100",
            "results": [
                {
                    "title": "Mozilla Otari Gateway Documentation — Per-User Budgets & Reservations",
                    "url": "https://docs.otari.ai/budgets",
                    "snippet": (
                        "Otari reserves worst-case estimated cost upfront before calling the provider "
                        "and reconciles actual spend upon completion."
                    ),
                },
                {
                    "title": "Mozilla Otari Issue #2992 — Service-Key End-User Refusal Naming",
                    "url": "https://github.com/mozilla-ai/otari/issues/2992",
                    "snippet": (
                        "reserve_budget formats User 'eu_<hex>' instead of End user '<external_id>' "
                        "when a service key's end user exceeds its budget."
                    ),
                },
            ],
        }

    @app.post("/api/v1/fetch")
    async def web_fetch(request: Request) -> dict[str, Any]:
        body = await request.json()
        url = body.get("url", "https://docs.otari.ai/end-users")
        return {
            "url": url,
            "title": "Otari End-Users & Service Keys Specification",
            "content_type": "text/markdown",
            "markdown": (
                "# End-User Budgets on Service Keys\n"
                "Pass `user` in the request body and `Otari-End-User-Budget` in headers to cap "
                "each customer or autonomous subagent independently."
            ),
        }

    @app.post("/api/v1/code/execute")
    async def code_execute(request: Request) -> dict[str, Any]:
        body = await request.json()
        code = body.get("code", "")
        return {
            "language": body.get("language", "python"),
            "exit_code": 0,
            "stdout": (
                "pytest tests/unit/test_budget_refusal_detail.py: 8 passed in 0.42s\n"
                f"Verified snippet ({len(code)} bytes): End user 'alice' refusal wording OK."
            ),
            "stderr": "",
            "duration_ms": 42.5,
        }

    @app.get("/api/v1/mcp/servers")
    async def list_mcp_servers() -> list[dict[str, Any]]:
        return [
            {
                "id": "repo-inspector-mcp",
                "name": "Otari Repository & Issue Inspector MCP",
                "transport": "streamable_http",
                "tools": ["inspect_otari_issue", "verify_patch_diff"],
            },
            {
                "id": "finops-ledger-mcp",
                "name": "FinOps Budget Audit MCP",
                "transport": "streamable_http",
                "tools": ["audit_subagent_spend"],
            },
        ]

    @app.post("/api/v1/mcp/execute")
    async def execute_mcp_tool(request: Request) -> dict[str, Any]:
        body = await request.json()
        server_id = body.get("server_id", "repo-inspector-mcp")
        tool_name = body.get("tool_name", "inspect_otari_issue")
        arguments = body.get("arguments", {})
        return {
            "server_id": server_id,
            "tool_name": tool_name,
            "isError": False,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "issue": arguments.get("issue_number", 2992),
                            "file": "src/gateway/services/budgets/_reservations.py",
                            "lines": [800, 1035],
                            "root_cause": (
                                "Uses f\"User '{user_id}'\" instead of user.external_id for service-key end users"
                            ),
                        }
                    ),
                }
            ],
        }

    @app.post("/api/v1/chat/completions")
    @app.post("/api/v1/messages")
    async def chat_completions(
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        otari_end_user_budget: str | None = Header(default=None, alias="Otari-End-User-Budget"),
        otari_tags: str | None = Header(default=None, alias="Otari-Tags"),
        otari_web_search: str | None = Header(default=None, alias="Otari-Web-Search"),
        otari_web_fetch: str | None = Header(default=None, alias="Otari-Web-Fetch"),
        otari_code_execution: str | None = Header(default=None, alias="Otari-Code-Execution"),
    ) -> JSONResponse:
        raw_body = await request.body()
        body_hash = hashlib.sha256(raw_body).hexdigest()
        body = json.loads(raw_body.decode("utf-8"))

        # 1. Check Idempotency-Key replay
        if idempotency_key:
            if idempotency_key in gw.idempotency_store:
                stored_hash, stored_payload, stored_headers = gw.idempotency_store[idempotency_key]
                if stored_hash != body_hash:
                    raise HTTPException(
                        status_code=422,
                        detail="Idempotency-Key was already used with a different request payload",
                        headers={"Otari-Error-Code": "idempotency_key_reused"},
                    )
                replay_headers = dict(stored_headers)
                replay_headers["Otari-Idempotent-Replayed"] = "true"
                return JSONResponse(status_code=200, content=stored_payload, headers=replay_headers)

        # 2. Resolve service-key end user
        external_user = body.get("user") or (body.get("metadata") or {}).get("user_id") or "default-swarm-user"
        end_user = gw.get_or_create_end_user(external_user, otari_end_user_budget)
        if end_user.blocked:
            subject = (
                f"User '{end_user.internal_user_id}'"
                if gw.simulate_issue_2992_internal_id
                else f"End user '{end_user.external_id}'"
            )
            return JSONResponse(
                status_code=403,
                content={"detail": f"{subject} is blocked", "code": "user_blocked"},
                headers={"Otari-Error-Code": "user_blocked"},
            )

        # 3. Evaluate per-request guardrails
        messages = body.get("messages", [])
        prompt_text = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        guardrail_rules = body.get("guardrails") or []
        guardrail_verdicts: list[dict[str, Any]] = []
        for rule in guardrail_rules:
            profile = rule.get("profile", "prompt-injection")
            mode = rule.get("mode", "block")
            triggered = any(
                marker in prompt_text.lower()
                for marker in (
                    "ignore previous instructions",
                    "exfiltrate",
                    "otari_master_key",
                    "drop table",
                )
            )
            verdict = {
                "profile": profile,
                "mode": mode,
                "triggered": triggered,
                "action": "blocked" if (triggered and mode == "block") else ("flagged" if triggered else "passed"),
            }
            guardrail_verdicts.append(verdict)
            if triggered and mode == "block":
                # Emulate Otari Issue #2989 when configured: nested detail.code without Otari-Error-Code header
                headers = {"Otari-Guardrails": json.dumps(guardrail_verdicts)}
                if not gw.simulate_issue_2989_missing_header:
                    headers["Otari-Error-Code"] = "guardrail_violation"
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": {
                            "error": f"Guardrail '{profile}' blocked the request",
                            "code": "guardrail_violation",
                            "profile": profile,
                        }
                    },
                    headers=headers,
                )

        # 4. Model resolution & simulated Issue #2981 provider-scoped ceiling
        requested_model = str(body.get("model", "together:Qwen/Qwen2.5-Coder-32B-Instruct"))
        if requested_model in gw.simulate_first_candidate_provider_ceiling_exhausted:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        f"Workspace has $0.000100 of budget remaining for provider "
                        f"'{requested_model.split(':', 1)[0]}', less than this request's estimated cost of $0.001200"
                    ),
                    "code": "budget_exceeded",
                },
                headers={
                    "Otari-Error-Code": "budget_exceeded",
                    "Otari-Budget-Scope": "workspace",
                },
            )

        provider, model_name = (
            requested_model.split(":", 1) if ":" in requested_model else ("together", requested_model)
        )
        in_rate, out_rate = MODEL_PRICING_PER_1K.get(
            requested_model,
            MODEL_PRICING_PER_1K.get(model_name, (Decimal("0.00020"), Decimal("0.00060"))),
        )

        # 5. Upfront worst-case budget hold check (mirroring Otari's reserve_budget)
        input_tokens = max(120, min(1200, len(prompt_text) // 3 + 150))
        max_output_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        estimated_hold = (
            (Decimal(input_tokens) / Decimal(1000)) * in_rate + (Decimal(max_output_tokens) / Decimal(1000)) * out_rate
        ).quantize(Decimal("0.000001"))

        remaining_before = max(Decimal("0"), end_user.max_budget - (end_user.spend + end_user.reserved))
        if end_user.spend + end_user.reserved + estimated_hold > end_user.max_budget:
            subject = (
                f"User '{end_user.internal_user_id}'"
                if gw.simulate_issue_2992_internal_id
                else f"End user '{end_user.external_id}'"
            )
            if remaining_before > Decimal("0"):
                detail = (
                    f"{subject} has ${remaining_before:.6f} of budget remaining, "
                    f"less than this request's estimated cost of ${estimated_hold:.6f}. "
                    f"The estimate assumes {max_output_tokens} output tokens; "
                    "a lower output token limit on the request may let it fit"
                )
            else:
                detail = f"{subject} has exceeded budget limit"
            return JSONResponse(
                status_code=403,
                content={"detail": detail, "code": "budget_exceeded"},
                headers={
                    "Otari-Error-Code": "budget_exceeded",
                    "Otari-Budget-Scope": "user",
                    "Retry-After": "86400",
                },
            )

        # 6. Execute gateway built-in tools & MCP if requested
        tool_calls: list[OtariToolCall] = []
        if otari_web_search == "otari":
            tool_calls.append(
                OtariToolCall(
                    id=f"tc_search_{uuid.uuid4().hex[:8]}",
                    type="web_search",
                    name="otari_web_search",
                    arguments={"query": "mozilla-ai/otari budget reservation end-user external_id"},
                    status="completed",
                    output_preview="Found 2 relevant docs: docs/budgets.md and Issue #2992",
                    duration_ms=18.4,
                )
            )
        if otari_web_fetch == "otari":
            tool_calls.append(
                OtariToolCall(
                    id=f"tc_fetch_{uuid.uuid4().hex[:8]}",
                    type="web_fetch",
                    name="otari_web_fetch",
                    arguments={"url": "https://docs.otari.ai/end-users"},
                    status="completed",
                    output_preview="Fetched 1.8 KB markdown from docs/end-users.md",
                    duration_ms=14.2,
                )
            )
        if otari_code_execution == "otari":
            tool_calls.append(
                OtariToolCall(
                    id=f"tc_exec_{uuid.uuid4().hex[:8]}",
                    type="code_execution",
                    name="otari_code_execution",
                    arguments={"language": "python", "code": "pytest tests/unit/test_budget_refusal_detail.py"},
                    status="completed",
                    output_preview="8 passed in 0.42s (exit_code=0)",
                    duration_ms=42.5,
                )
            )
        for mcp_srv in body.get("mcp_servers") or []:
            srv_id = mcp_srv if isinstance(mcp_srv, str) else mcp_srv.get("id", "repo-inspector-mcp")
            tool_calls.append(
                OtariToolCall(
                    id=f"tc_mcp_{uuid.uuid4().hex[:8]}",
                    type="mcp",
                    name=f"{srv_id}:inspect_otari_issue",
                    arguments={"issue_number": 2992},
                    status="completed",
                    output_preview="Confirmed _reservations.py lines 800 & 1035 use internal user_id",
                    duration_ms=21.7,
                )
            )

        # 7. Reconcile actual token usage & cost (typically 55-70% of max_output_tokens)
        actual_output_tokens = max(96, min(max_output_tokens, int(max_output_tokens * 0.65)))
        actual_cost = (
            (Decimal(input_tokens) / Decimal(1000)) * in_rate
            + (Decimal(actual_output_tokens) / Decimal(1000)) * out_rate
        ).quantize(Decimal("0.000001"))

        end_user.spend += actual_cost
        end_user.current_tokens += input_tokens + actual_output_tokens
        end_user.current_requests += 1

        gw.key_spend_usd += actual_cost
        gw.key_used_tokens += input_tokens + actual_output_tokens
        gw.key_used_requests += 1

        # Record tag usage bucket
        role_tag = "untagged"
        if otari_tags:
            for part in otari_tags.split(","):
                if "=" in part:
                    k, v = part.strip().split("=", 1)
                    if k.strip() == "role":
                        role_tag = v.strip()
        bucket_key = (role_tag, provider, model_name)
        bucket = gw.usage_buckets.get(bucket_key)
        if bucket is None:
            bucket = UsageSummaryBucket(tag=role_tag, provider=provider, model=model_name)
            gw.usage_buckets[bucket_key] = bucket
        bucket.requests += 1
        bucket.input_tokens += input_tokens
        bucket.output_tokens += actual_output_tokens
        bucket.total_tokens += input_tokens + actual_output_tokens
        bucket.cost_usd += actual_cost

        attempt_count = 1
        fallback_used = False
        if requested_model in gw.simulate_transient_429_once_for_models:
            gw.simulate_transient_429_once_for_models.remove(requested_model)
            attempt_count = 2
            fallback_used = True

        req_id = f"req_{uuid.uuid4().hex[:16]}"
        response_headers = {
            "Otari-Request-ID": req_id,
            "Otari-Provider": provider,
            "Otari-Model": model_name,
            "Otari-Attempt-Count": str(attempt_count),
            "Otari-Fallback": "true" if fallback_used else "false",
            "Otari-Response-Duration-Ms": "68",
            "Otari-End-User-Budget": end_user.budget_id,
            "Server-Timing": "gateway;dur=4.2, tool;dur=19.8, upstream;dur=44.0",
        }
        if guardrail_verdicts:
            response_headers["Otari-Guardrails"] = json.dumps(guardrail_verdicts)

        content_summary = (
            f"[{provider}:{model_name}] Completed step for end_user='{end_user.external_id}' "
            f"(actual_cost=${actual_cost:.6f}, output_tokens={actual_output_tokens}/{max_output_tokens})."
        )
        response_payload: dict[str, Any] = {
            "id": f"chatcmpl-{req_id}",
            "object": "chat.completion",
            "model": model_name,
            "provider": provider,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content_summary,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": actual_output_tokens,
                "total_tokens": input_tokens + actual_output_tokens,
                "cost_usd": float(actual_cost),
            },
            "otari_tool_calls": [tc.model_dump() for tc in tool_calls],
        }

        if idempotency_key:
            gw.idempotency_store[idempotency_key] = (body_hash, response_payload, response_headers)

        return JSONResponse(status_code=200, content=response_payload, headers=response_headers)

    return app
