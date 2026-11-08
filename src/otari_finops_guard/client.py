"""Mozilla Otari FinOps Gateway Client & Refusal Normalizer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx

from otari_finops_guard.models import (
    EndUserBudgetPublic,
    EndUserPublic,
    GuardrailRule,
    KeyBudgetState,
    OtariToolCall,
    UsageSummaryResponse,
)

_INTERNAL_EU_RE = re.compile(r"User '(eu_[0-9a-f]{16,64})'")


@dataclass
class NormalizedOtariError(Exception):
    """Structured error raised when Otari refuses a request (403/409/422/429)."""

    status_code: int
    error_code: str
    raw_detail: str
    normalized_detail: str
    budget_scope: str | None = None
    retry_after: int | None = None
    guardrails_header: str = ""
    repaired_issue: str | None = None


def normalize_otari_refusal(
    response: httpx.Response,
    *,
    external_user_id: str | None = None,
) -> NormalizedOtariError:
    """Normalize an Otari error response, repairing upstream Issues #2992 and #2989."""
    try:
        payload = response.json()
    except Exception:
        payload = {"detail": response.text}

    raw_detail_obj = payload.get("detail", payload) if isinstance(payload, dict) else str(payload)
    header_code = response.headers.get("Otari-Error-Code")
    budget_scope = response.headers.get("Otari-Budget-Scope")
    retry_after_raw = response.headers.get("Retry-After")
    retry_after = int(retry_after_raw) if (retry_after_raw and retry_after_raw.isdigit()) else None
    guardrails_header = response.headers.get("Otari-Guardrails", "")
    repaired_issue: str | None = None

    # Handle Issue #2989: guardrail_violation 403 nests {"code": "guardrail_violation"} inside detail
    # without setting Otari-Error-Code header or top-level code.
    if isinstance(raw_detail_obj, dict):
        nested_code = raw_detail_obj.get("code")
        raw_text = str(raw_detail_obj.get("error") or raw_detail_obj)
        error_code = header_code or (str(nested_code) if nested_code else "unknown_error")
        if header_code is None and nested_code == "guardrail_violation":
            repaired_issue = "mozilla-ai/otari#2989"
    else:
        raw_text = str(raw_detail_obj)
        top_code = payload.get("code") if isinstance(payload, dict) else None
        error_code = header_code or (str(top_code) if top_code else "unknown_error")

    # Handle Issue #2992: reserve_budget formats User 'eu_<hex>' instead of End user '<external_id>'
    normalized_text = raw_text
    if external_user_id and _INTERNAL_EU_RE.search(raw_text):
        normalized_text = _INTERNAL_EU_RE.sub(f"End user '{external_user_id}'", raw_text)
        repaired_issue = "mozilla-ai/otari#2992"

    return NormalizedOtariError(
        status_code=response.status_code,
        error_code=error_code,
        raw_detail=raw_text,
        normalized_detail=normalized_text,
        budget_scope=budget_scope,
        retry_after=retry_after,
        guardrails_header=guardrails_header,
        repaired_issue=repaired_issue,
    )


@dataclass
class OtariCompletionResult:
    """Parsed completion response + Otari telemetry headers."""

    request_id: str
    provider: str
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    attempt_count: int
    fallback_used: bool
    idempotent_replayed: bool
    end_user_budget_id: str
    guardrails_header: str
    server_timing: str
    tool_calls: list[OtariToolCall]


class OtariFinOpsClient:
    """Async client for Mozilla Otari's inference, FinOps, routing, tool, and MCP endpoints."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        api_key: str = "otari_sk_service_demo_key",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Otari-Key": self.api_key,
            },
            transport=transport,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OtariFinOpsClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def get_key_budget(self) -> KeyBudgetState:
        """Inspect the virtual/service key's own remaining budget (GET /api/v1/key/budget)."""
        resp = await self._client.get("/api/v1/key/budget")
        resp.raise_for_status()
        return KeyBudgetState.model_validate(resp.json())

    async def get_usage_summary(self, *, group_by_tag: str = "role") -> UsageSummaryResponse:
        """Fetch spend breakdown grouped by tag (GET /api/v1/key/usage/summary)."""
        resp = await self._client.get(
            "/api/v1/key/usage/summary",
            params={"group_by_tag": group_by_tag, "dimensions": "none"},
        )
        resp.raise_for_status()
        return UsageSummaryResponse.model_validate(resp.json())

    async def list_end_user_budgets(self) -> list[EndUserBudgetPublic]:
        """List budgets assignable by this service key (GET /api/v1/end-user-budgets)."""
        resp = await self._client.get("/api/v1/end-user-budgets")
        resp.raise_for_status()
        return [EndUserBudgetPublic.model_validate(item) for item in resp.json()]

    async def provision_end_user(self, external_id: str, budget_id: str) -> EndUserPublic:
        """Create or assign a subagent/end-user onto a specific budget (PUT /api/v1/end-users)."""
        resp = await self._client.put(
            "/api/v1/end-users",
            params={"external_id": external_id},
            json={"budget_id": budget_id},
        )
        if resp.status_code >= 400:
            raise normalize_otari_refusal(resp, external_user_id=external_id)
        return EndUserPublic.model_validate(resp.json())

    async def get_end_user(self, external_id: str) -> EndUserPublic:
        """Inspect a subagent/end-user's current spend and remaining budget (GET /api/v1/end-users)."""
        resp = await self._client.get("/api/v1/end-users", params={"external_id": external_id})
        if resp.status_code >= 400:
            raise normalize_otari_refusal(resp, external_user_id=external_id)
        return EndUserPublic.model_validate(resp.json())

    async def list_end_users(self) -> list[EndUserPublic]:
        """List all end-users managed by this service key."""
        resp = await self._client.get("/api/v1/end-users")
        resp.raise_for_status()
        data = resp.json()
        items = data if isinstance(data, list) else data.get("items", [])
        return [EndUserPublic.model_validate(u) for u in items]

    async def recommend_subagent_models(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Query Otari's subagent model recommender (POST /api/v1/routing/recommend)."""
        resp = await self._client.post("/api/v1/routing/recommend", json=payload)
        resp.raise_for_status()
        return dict(resp.json())

    async def web_search(self, query: str) -> dict[str, Any]:
        """Call Otari's built-in web search endpoint (POST /api/v1/search)."""
        resp = await self._client.post("/api/v1/search", json={"query": query})
        resp.raise_for_status()
        return dict(resp.json())

    async def web_fetch(self, url: str) -> dict[str, Any]:
        """Call Otari's built-in web fetch endpoint (POST /api/v1/fetch)."""
        resp = await self._client.post("/api/v1/fetch", json={"url": url})
        resp.raise_for_status()
        return dict(resp.json())

    async def code_execute(self, code: str, *, language: str = "python") -> dict[str, Any]:
        """Call Otari's sandboxed code execution endpoint (POST /api/v1/code/execute)."""
        resp = await self._client.post(
            "/api/v1/code/execute",
            json={"language": language, "code": code},
        )
        resp.raise_for_status()
        return dict(resp.json())

    async def list_mcp_servers(self) -> list[dict[str, Any]]:
        """List MCP servers configured on the Otari gateway (GET /api/v1/mcp/servers)."""
        resp = await self._client.get("/api/v1/mcp/servers")
        resp.raise_for_status()
        return list(resp.json())

    async def execute_mcp_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute an MCP tool through the Otari gateway (POST /api/v1/mcp/execute)."""
        resp = await self._client.post(
            "/api/v1/mcp/execute",
            json={
                "server_id": server_id,
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )
        resp.raise_for_status()
        return dict(resp.json())

    async def chat_completion(
        self,
        *,
        model: str,
        fallback_model: str | None = None,
        messages: list[dict[str, str]],
        end_user_id: str,
        end_user_budget_id: str,
        role_tag: str,
        task_id: str,
        max_tokens: int = 1024,
        idempotency_key: str | None = None,
        enable_web_search: bool = False,
        enable_web_fetch: bool = False,
        enable_code_execution: bool = False,
        mcp_servers: list[str] | None = None,
        guardrails: list[GuardrailRule] | None = None,
    ) -> OtariCompletionResult:
        """Send an OpenAI-compatible chat completion through Mozilla Otari with FinOps headers.

        Includes client-side mitigation for Issue #2981: if the primary candidate
        fails with a workspace/organization provider-scoped budget_exceeded and a
        fallback_model on another provider is available, automatically retries on
        fallback_model.
        """
        headers: dict[str, str] = {
            "Otari-End-User-Budget": end_user_budget_id,
            "Otari-Tags": f"role={role_tag},task={task_id}",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if enable_web_search:
            headers["Otari-Web-Search"] = "otari"
        if enable_web_fetch:
            headers["Otari-Web-Fetch"] = "otari"
        if enable_code_execution:
            headers["Otari-Code-Execution"] = "otari"

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "user": end_user_id,
            "max_tokens": max_tokens,
        }
        if mcp_servers:
            body["mcp_servers"] = mcp_servers
        if guardrails:
            body["guardrails"] = [g.model_dump() for g in guardrails]

        resp = await self._client.post("/api/v1/chat/completions", json=body, headers=headers)
        if resp.status_code >= 400:
            err = normalize_otari_refusal(resp, external_user_id=end_user_id)
            # Issue #2981 workaround: if a routing policy's primary candidate hits a
            # provider-scoped workspace/organization ceiling, retry on the fallback candidate.
            if (
                err.error_code == "budget_exceeded"
                and err.budget_scope in ("workspace", "organization")
                and fallback_model
                and fallback_model != model
            ):
                body["model"] = fallback_model
                retry_resp = await self._client.post(
                    "/api/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                if retry_resp.status_code < 400:
                    result = self._parse_completion(retry_resp)
                    result.fallback_used = True
                    result.attempt_count += 1
                    return result
                raise normalize_otari_refusal(retry_resp, external_user_id=end_user_id)
            raise err

        return self._parse_completion(resp)

    def _parse_completion(self, resp: httpx.Response) -> OtariCompletionResult:
        data = resp.json()
        usage = data.get("usage", {})
        choices = data.get("choices", [{}])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        tool_calls = [OtariToolCall.model_validate(tc) for tc in (data.get("otari_tool_calls") or [])]
        return OtariCompletionResult(
            request_id=resp.headers.get("Otari-Request-ID", data.get("id", "")),
            provider=resp.headers.get("Otari-Provider", data.get("provider", "unknown")),
            model=resp.headers.get("Otari-Model", data.get("model", "unknown")),
            content=content,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cost_usd=Decimal(str(usage.get("cost_usd", "0"))).quantize(Decimal("0.000001")),
            attempt_count=int(resp.headers.get("Otari-Attempt-Count", "1")),
            fallback_used=resp.headers.get("Otari-Fallback", "false").lower() == "true",
            idempotent_replayed=resp.headers.get("Otari-Idempotent-Replayed", "false").lower() == "true",
            end_user_budget_id=resp.headers.get("Otari-End-User-Budget", ""),
            guardrails_header=resp.headers.get("Otari-Guardrails", ""),
            server_timing=resp.headers.get("Server-Timing", ""),
            tool_calls=tool_calls,
        )
