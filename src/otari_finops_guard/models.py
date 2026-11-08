"""Domain models and Mozilla Otari wire schemas for Otari FinOps Guard."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentRole(StrEnum):
    """Specialized subagent roles in a multi-agent coding swarm."""

    ARCHITECT = "architect"
    RESEARCHER = "researcher"
    CODER = "coder"
    VERIFIER = "verifier"


class BudgetWatermarkTier(StrEnum):
    """Budget utilization watermark states for a virtual key or service-key end user."""

    OPTIMAL = "optimal"  # 0% - <50%: Full primary model tier + standard max_tokens
    RIGHT_SIZED = "right_sized"  # 50% - <75%: Primary model tier + dynamic max_tokens clamping
    DEGRADED_OPEN_WEIGHT = "degraded_open_weight"  # 75% - <100%: Downgraded to fast open-weight model
    HALTED_CHECKPOINTED = "halted_checkpointed"  # >=100% (or insufficient hold room): Halted & checkpointed


class ModelCandidate(BaseModel):
    """A model candidate registered with Otari FinOps Guard."""

    provider: str
    model: str
    open_weight: bool = True
    role: AgentRole
    tier: str = Field(description="'primary' or 'economy_fallback'")
    input_cost_per_1k: Decimal
    output_cost_per_1k: Decimal
    default_max_tokens: int = 2048
    min_viable_max_tokens: int = 128
    description: str

    @property
    def target(self) -> str:
        """Otari 'provider:model' target identifier."""
        return f"{self.provider}:{self.model}"

    def estimate_hold_usd(self, input_tokens: int, max_output_tokens: int) -> Decimal:
        """Estimate the worst-case upfront hold Otari's reserve_budget computes."""
        in_cost = (Decimal(input_tokens) / Decimal(1000)) * self.input_cost_per_1k
        out_cost = (Decimal(max_output_tokens) / Decimal(1000)) * self.output_cost_per_1k
        return (in_cost + out_cost).quantize(Decimal("0.000001"))


class KeyBudgetLimit(BaseModel):
    """One active limit from GET /api/v1/key/budget."""

    scope: str = "user"
    provider: str | None = None
    max_budget_usd: Decimal
    spend_usd: Decimal
    reserved_usd: Decimal = Decimal("0")
    remaining_usd: Decimal
    max_tokens: int | None = None
    used_tokens: int = 0
    reserved_tokens: int = 0
    remaining_tokens: int | None = None
    max_requests: int | None = None
    used_requests: int = 0
    reserved_requests: int = 0
    remaining_requests: int | None = None
    budget_duration: str | None = "1d"
    resets_at: str | None = None


class KeyBudgetState(BaseModel):
    """Response from GET /api/v1/key/budget."""

    budgets_enforced: bool = True
    remaining_usd: Decimal | None = None
    remaining_tokens: int | None = None
    remaining_requests: int | None = None
    resets_at: str | None = None
    limits: list[KeyBudgetLimit] = Field(default_factory=list)


class EndUserBudgetPublic(BaseModel):
    """One assignable end-user budget from GET /api/v1/end-user-budgets."""

    budget_id: str
    max_budget: Decimal | None = None
    budget_duration: str | None = "1d"
    token_limit: int | None = None
    request_limit: int | None = None
    is_default: bool = False


class EndUserPublic(BaseModel):
    """End-user state from GET/PUT/PATCH /api/v1/end-users."""

    external_id: str
    budget_id: str | None = None
    max_budget: Decimal | None = None
    spend: Decimal = Decimal("0")
    reserved: Decimal = Decimal("0")
    remaining_budget: Decimal | None = None
    token_limit: int | None = None
    current_tokens: int = 0
    reserved_tokens: int = 0
    request_limit: int | None = None
    current_requests: int = 0
    reserved_requests: int = 0
    budget_duration: str | None = "1d"
    next_budget_reset_at: str | None = None
    blocked: bool = False

    @property
    def utilization_ratio(self) -> float:
        """Fraction of max_budget consumed + reserved (0.0 to 1.0+)."""
        if not self.max_budget or self.max_budget <= Decimal("0"):
            return 0.0
        committed = self.spend + self.reserved
        return float(committed / self.max_budget)


class UsageSummaryBucket(BaseModel):
    """A single tag/model bucket from GET /api/v1/key/usage/summary."""

    tag: str | None = None
    provider: str | None = None
    model: str | None = None
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Decimal = Decimal("0")


class UsageSummaryResponse(BaseModel):
    """Response from GET /api/v1/key/usage/summary."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: Decimal = Decimal("0")
    buckets: list[UsageSummaryBucket] = Field(default_factory=list)


class GuardrailRule(BaseModel):
    """Per-request guardrail rule passed in the Otari request body."""

    profile: str
    mode: str = "block"  # "block" | "monitor" | "redact"
    on: str = "input"  # "input" | "output" | "both"


class OtariToolCall(BaseModel):
    """Gateway-executed built-in tool or MCP call reported in otari_tool_calls."""

    id: str
    type: str  # "web_search" | "web_fetch" | "code_execution" | "mcp"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    output_preview: str = ""
    duration_ms: float = 0.0


class RoutingDecision(BaseModel):
    """Decision emitted by the Adaptive FinOps Model Router before calling Otari."""

    role: AgentRole
    end_user_id: str
    budget_id: str
    watermark_tier: BudgetWatermarkTier
    utilization_pct: float
    remaining_usd: Decimal
    selected_candidate: ModelCandidate
    fallback_candidate: ModelCandidate | None = None
    downgraded_from: str | None = None
    requested_max_tokens: int
    clamped_max_tokens: int
    estimated_hold_usd: Decimal
    rationale: str


class TelemetryEvent(BaseModel):
    """Rich telemetry record for every swarm step and Otari gateway interaction."""

    step_index: int
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    task_id: str
    subagent_role: AgentRole
    end_user_id: str
    budget_id: str
    watermark_tier: BudgetWatermarkTier
    utilization_before_pct: float
    utilization_after_pct: float
    provider: str
    model: str
    open_weight: bool = True
    downgraded_from: str | None = None
    clamped_max_tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    remaining_usd_after: Decimal = Decimal("0")
    otari_request_id: str = ""
    otari_attempt_count: int = 1
    otari_fallback: bool = False
    otari_idempotent_replayed: bool = False
    otari_guardrails_header: str = ""
    otari_error_code: str | None = None
    otari_budget_scope: str | None = None
    normalized_refusal_detail: str | None = None
    tool_calls: list[OtariToolCall] = Field(default_factory=list)
    status: str = "completed"  # "completed" | "downgraded" | "guardrail_blocked" | "checkpointed"
    summary: str = ""


class SwarmTaskCheckpoint(BaseModel):
    """Deterministic state checkpoint written when a subagent hits 100% of its Otari budget."""

    checkpoint_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    task_id: str
    halted_subagent_role: AgentRole
    end_user_id: str
    budget_id: str
    max_budget_usd: Decimal
    spend_usd: Decimal
    remaining_usd: Decimal
    utilization_pct: float
    required_hold_usd: Decimal
    next_budget_reset_at: str | None = None
    last_idempotency_key: str
    completed_steps: list[str] = Field(default_factory=list)
    pending_steps: list[str] = Field(default_factory=list)
    partial_artifacts: dict[str, str] = Field(default_factory=dict)
    resume_command: str


class SwarmRunReport(BaseModel):
    """Aggregate report of a multi-agent coding swarm execution through Otari FinOps Guard."""

    run_id: str
    started_at: str
    completed_at: str
    gateway_base_url: str
    gateway_mode: str  # "mock_asgi" | "live_http"
    total_requests: int
    total_cost_usd: Decimal
    open_weight_models_used: list[str]
    otari_features_exercised: list[str]
    tier_downgrades_count: int
    max_tokens_clamps_count: int
    guardrail_blocks_count: int
    idempotent_replays_count: int
    end_users: list[EndUserPublic]
    usage_by_role: list[UsageSummaryBucket]
    events: list[TelemetryEvent]
    checkpoint: SwarmTaskCheckpoint | None = None
    upstream_issues_verified: list[dict[str, str]] = Field(default_factory=list)
