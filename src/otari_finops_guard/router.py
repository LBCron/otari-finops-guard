"""Adaptive FinOps Model Router for Mozilla Otari.

Maps specialized multi-agent coding roles onto open-weight models, enforces
75% budget watermark tier downgrades, right-sizes max_tokens so Otari's upfront
worst-case reservation hold fits inside remaining headroom, and triggers
clean 100% deterministic checkpointing.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal

from otari_finops_guard.models import (
    AgentRole,
    BudgetWatermarkTier,
    EndUserPublic,
    ModelCandidate,
    RoutingDecision,
)

# Default Open-Weight Model Catalog across 4 distinct agent roles
OPEN_WEIGHT_CATALOG: dict[AgentRole, tuple[ModelCandidate, ModelCandidate]] = {
    AgentRole.ARCHITECT: (
        ModelCandidate(
            provider="deepseek",
            model="deepseek-r1",
            open_weight=True,
            role=AgentRole.ARCHITECT,
            tier="primary",
            input_cost_per_1k=Decimal("0.00055"),
            output_cost_per_1k=Decimal("0.00219"),
            default_max_tokens=2048,
            min_viable_max_tokens=160,
            description=(
                "DeepSeek-R1 (671B MoE open-weight reasoning model) for multi-step system architecture, "
                "dependency graph planning, and root-cause hypothesis generation."
            ),
        ),
        ModelCandidate(
            provider="together",
            model="meta-llama/Llama-3.3-70B-Instruct",
            open_weight=True,
            role=AgentRole.ARCHITECT,
            tier="economy_fallback",
            input_cost_per_1k=Decimal("0.00018"),
            output_cost_per_1k=Decimal("0.00054"),
            default_max_tokens=1024,
            min_viable_max_tokens=128,
            description=(
                "Llama-3.3-70B-Instruct open-weight fallback for concise architectural planning "
                "when subagent budget passes the 75% watermark."
            ),
        ),
    ),
    AgentRole.RESEARCHER: (
        ModelCandidate(
            provider="together",
            model="meta-llama/Llama-3.3-70B-Instruct",
            open_weight=True,
            role=AgentRole.RESEARCHER,
            tier="primary",
            input_cost_per_1k=Decimal("0.00018"),
            output_cost_per_1k=Decimal("0.00054"),
            default_max_tokens=1536,
            min_viable_max_tokens=128,
            description=(
                "Llama-3.3-70B-Instruct open-weight generalist for synthesizing Otari web search, "
                "web fetch, and documentation retrieval results."
            ),
        ),
        ModelCandidate(
            provider="mistral",
            model="mistral-small-24b-instruct-2501",
            open_weight=True,
            role=AgentRole.RESEARCHER,
            tier="economy_fallback",
            input_cost_per_1k=Decimal("0.00007"),
            output_cost_per_1k=Decimal("0.00020"),
            default_max_tokens=1024,
            min_viable_max_tokens=128,
            description=(
                "Mistral-Small-24B-Instruct-2501 open-weight fast model for low-cost retrieval "
                "summarization above the 75% budget watermark."
            ),
        ),
    ),
    AgentRole.CODER: (
        ModelCandidate(
            provider="together",
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            open_weight=True,
            role=AgentRole.CODER,
            tier="primary",
            input_cost_per_1k=Decimal("0.00020"),
            output_cost_per_1k=Decimal("0.00060"),
            default_max_tokens=2048,
            min_viable_max_tokens=160,
            description=(
                "Qwen2.5-Coder-32B-Instruct open-weight code specialist for surgical patch generation, "
                "MCP repository inspection, and unit test authoring."
            ),
        ),
        ModelCandidate(
            provider="mistral",
            model="mistral-small-24b-instruct-2501",
            open_weight=True,
            role=AgentRole.CODER,
            tier="economy_fallback",
            input_cost_per_1k=Decimal("0.00007"),
            output_cost_per_1k=Decimal("0.00020"),
            default_max_tokens=1024,
            min_viable_max_tokens=128,
            description=(
                "Mistral-Small-24B-Instruct-2501 open-weight economy coder fallback when a subagent "
                "crosses 75% of its Otari budget."
            ),
        ),
    ),
    AgentRole.VERIFIER: (
        ModelCandidate(
            provider="mistral",
            model="mistral-small-24b-instruct-2501",
            open_weight=True,
            role=AgentRole.VERIFIER,
            tier="primary",
            input_cost_per_1k=Decimal("0.00007"),
            output_cost_per_1k=Decimal("0.00020"),
            default_max_tokens=1024,
            min_viable_max_tokens=128,
            description=(
                "Mistral-Small-24B-Instruct-2501 open-weight verifier for fast sandbox output validation, "
                "diff auditing, and guardrail compliance checks."
            ),
        ),
        ModelCandidate(
            provider="mistral",
            model="mistral-small-24b-instruct-2501",
            open_weight=True,
            role=AgentRole.VERIFIER,
            tier="economy_fallback",
            input_cost_per_1k=Decimal("0.00007"),
            output_cost_per_1k=Decimal("0.00020"),
            default_max_tokens=512,
            min_viable_max_tokens=96,
            description=(
                "Mistral-Small-24B-Instruct-2501 compact verification mode with clamped output token "
                "allowance for tight remaining budget headroom."
            ),
        ),
    ),
}


class AdaptiveFinOpsRouter:
    """Budget-aware model router and max_tokens hold right-sizer for Mozilla Otari."""

    def __init__(
        self,
        *,
        right_size_watermark: float = 0.50,
        downgrade_watermark: float = 0.75,
        halt_watermark: float = 1.00,
        catalog: dict[AgentRole, tuple[ModelCandidate, ModelCandidate]] | None = None,
    ) -> None:
        self.right_size_watermark = right_size_watermark
        self.downgrade_watermark = downgrade_watermark
        self.halt_watermark = halt_watermark
        self.catalog = catalog or OPEN_WEIGHT_CATALOG

    def recommend_candidates_payload(self) -> dict[str, object]:
        """Build the POST /api/v1/routing/recommend payload for Otari's 'pi' subagent harness."""
        candidates: dict[str, str] = {}
        for role, (primary, fallback) in self.catalog.items():
            candidates[primary.target] = f"[{role.value}:primary] {primary.description}"
            if fallback.target not in candidates:
                candidates[fallback.target] = f"[{role.value}:fallback] {fallback.description}"
        return {
            "harness": "pi",
            "candidates": candidates,
        }

    def clamp_max_tokens_to_fit_hold(
        self,
        candidate: ModelCandidate,
        *,
        estimated_input_tokens: int,
        requested_max_tokens: int,
        remaining_usd: Decimal,
    ) -> int:
        """Right-size max_tokens so Otari's worst-case upfront hold fits inside remaining_usd.

        Otari's reserve_budget prices the upfront hold at:
          hold = (input_tokens * input_rate) + (max_tokens * output_rate)
        and refuses with 403 budget_exceeded ("The estimate assumes N output tokens;
        a lower output token limit on the request may let it fit") whenever hold > remaining_usd.
        """
        if remaining_usd <= Decimal("0"):
            return 0
        input_cost = (Decimal(estimated_input_tokens) / Decimal(1000)) * candidate.input_cost_per_1k
        # Keep a 2% safety buffer for tokenizer variance
        usable_for_output = (remaining_usd * Decimal("0.98")) - input_cost
        if usable_for_output <= Decimal("0"):
            return 0
        if candidate.output_cost_per_1k <= Decimal("0"):
            return requested_max_tokens
        affordable_tokens = int(
            ((usable_for_output / candidate.output_cost_per_1k) * Decimal(1000)).to_integral_value(rounding=ROUND_FLOOR)
        )
        return max(0, min(requested_max_tokens, affordable_tokens))

    def route(
        self,
        role: AgentRole,
        end_user: EndUserPublic,
        *,
        estimated_input_tokens: int = 450,
        requested_max_tokens: int | None = None,
    ) -> RoutingDecision:
        """Compute the optimal open-weight model and max_tokens for a subagent step."""
        primary, economy = self.catalog[role]
        max_budget = end_user.max_budget or Decimal("0.01")
        committed = end_user.spend + end_user.reserved
        remaining = max(Decimal("0"), max_budget - committed)
        ratio = float(committed / max_budget) if max_budget > Decimal("0") else 0.0
        utilization_pct = round(ratio * 100.0, 2)

        desired_tokens = requested_max_tokens or primary.default_max_tokens

        # 1. Check 100% hard cap
        if ratio >= self.halt_watermark or remaining <= Decimal("0"):
            min_hold = economy.estimate_hold_usd(estimated_input_tokens, economy.min_viable_max_tokens)
            return RoutingDecision(
                role=role,
                end_user_id=end_user.external_id,
                budget_id=end_user.budget_id or "default",
                watermark_tier=BudgetWatermarkTier.HALTED_CHECKPOINTED,
                utilization_pct=utilization_pct,
                remaining_usd=remaining,
                selected_candidate=economy,
                fallback_candidate=None,
                downgraded_from=primary.target,
                requested_max_tokens=desired_tokens,
                clamped_max_tokens=0,
                estimated_hold_usd=min_hold,
                rationale=(
                    f"End user '{end_user.external_id}' reached {utilization_pct:.1f}% of its "
                    f"${max_budget:.6f} Otari budget (${remaining:.6f} left). Halting and checkpointing state."
                ),
            )

        # 2. Check >= 75% watermark -> automatic downgrade to economy open-weight model
        if ratio >= self.downgrade_watermark:
            econ_desired = min(desired_tokens, economy.default_max_tokens)
            clamped = self.clamp_max_tokens_to_fit_hold(
                economy,
                estimated_input_tokens=estimated_input_tokens,
                requested_max_tokens=econ_desired,
                remaining_usd=remaining,
            )
            if clamped < economy.min_viable_max_tokens:
                min_hold = economy.estimate_hold_usd(estimated_input_tokens, economy.min_viable_max_tokens)
                return RoutingDecision(
                    role=role,
                    end_user_id=end_user.external_id,
                    budget_id=end_user.budget_id or "default",
                    watermark_tier=BudgetWatermarkTier.HALTED_CHECKPOINTED,
                    utilization_pct=utilization_pct,
                    remaining_usd=remaining,
                    selected_candidate=economy,
                    fallback_candidate=None,
                    downgraded_from=primary.target,
                    requested_max_tokens=desired_tokens,
                    clamped_max_tokens=0,
                    estimated_hold_usd=min_hold,
                    rationale=(
                        f"Remaining headroom (${remaining:.6f}) cannot fit minimum viable hold "
                        f"(${min_hold:.6f} for {economy.min_viable_max_tokens} tokens on {economy.target}). "
                        "Halting cleanly and writing checkpoint."
                    ),
                )
            hold = economy.estimate_hold_usd(estimated_input_tokens, clamped)
            downgraded_from = primary.target if economy.target != primary.target else None
            return RoutingDecision(
                role=role,
                end_user_id=end_user.external_id,
                budget_id=end_user.budget_id or "default",
                watermark_tier=BudgetWatermarkTier.DEGRADED_OPEN_WEIGHT,
                utilization_pct=utilization_pct,
                remaining_usd=remaining,
                selected_candidate=economy,
                fallback_candidate=None,
                downgraded_from=downgraded_from,
                requested_max_tokens=desired_tokens,
                clamped_max_tokens=clamped,
                estimated_hold_usd=hold,
                rationale=(
                    f"Budget utilization {utilization_pct:.1f}% >= 75% watermark: downgraded "
                    f"{primary.target} -> {economy.target} and right-sized max_tokens "
                    f"{desired_tokens} -> {clamped} (hold ${hold:.6f} <= ${remaining:.6f} remaining)."
                ),
            )

        # 3. Below 75%: try primary model first, clamping max_tokens if needed to fit upfront hold
        clamped_primary = self.clamp_max_tokens_to_fit_hold(
            primary,
            estimated_input_tokens=estimated_input_tokens,
            requested_max_tokens=desired_tokens,
            remaining_usd=remaining,
        )
        if clamped_primary < primary.min_viable_max_tokens:
            # Primary model's upfront hold won't fit in remaining headroom! Fall back to economy open-weight.
            clamped_econ = self.clamp_max_tokens_to_fit_hold(
                economy,
                estimated_input_tokens=estimated_input_tokens,
                requested_max_tokens=min(desired_tokens, economy.default_max_tokens),
                remaining_usd=remaining,
            )
            if clamped_econ < economy.min_viable_max_tokens:
                min_hold = economy.estimate_hold_usd(estimated_input_tokens, economy.min_viable_max_tokens)
                return RoutingDecision(
                    role=role,
                    end_user_id=end_user.external_id,
                    budget_id=end_user.budget_id or "default",
                    watermark_tier=BudgetWatermarkTier.HALTED_CHECKPOINTED,
                    utilization_pct=utilization_pct,
                    remaining_usd=remaining,
                    selected_candidate=economy,
                    fallback_candidate=None,
                    downgraded_from=primary.target,
                    requested_max_tokens=desired_tokens,
                    clamped_max_tokens=0,
                    estimated_hold_usd=min_hold,
                    rationale=(
                        f"Neither {primary.target} nor {economy.target} fits in ${remaining:.6f} "
                        "remaining headroom. Checkpointing before Otari 403 refusal."
                    ),
                )
            hold = economy.estimate_hold_usd(estimated_input_tokens, clamped_econ)
            return RoutingDecision(
                role=role,
                end_user_id=end_user.external_id,
                budget_id=end_user.budget_id or "default",
                watermark_tier=BudgetWatermarkTier.DEGRADED_OPEN_WEIGHT,
                utilization_pct=utilization_pct,
                remaining_usd=remaining,
                selected_candidate=economy,
                fallback_candidate=None,
                downgraded_from=primary.target,
                requested_max_tokens=desired_tokens,
                clamped_max_tokens=clamped_econ,
                estimated_hold_usd=hold,
                rationale=(
                    f"Primary {primary.target} worst-case hold exceeds ${remaining:.6f} remaining; "
                    f"proactively downgraded to {economy.target} with max_tokens={clamped_econ}."
                ),
            )

        hold = primary.estimate_hold_usd(estimated_input_tokens, clamped_primary)
        tier = (
            BudgetWatermarkTier.RIGHT_SIZED
            if (ratio >= self.right_size_watermark or clamped_primary < desired_tokens)
            else BudgetWatermarkTier.OPTIMAL
        )
        return RoutingDecision(
            role=role,
            end_user_id=end_user.external_id,
            budget_id=end_user.budget_id or "default",
            watermark_tier=tier,
            utilization_pct=utilization_pct,
            remaining_usd=remaining,
            selected_candidate=primary,
            fallback_candidate=economy if economy.target != primary.target else None,
            downgraded_from=None,
            requested_max_tokens=desired_tokens,
            clamped_max_tokens=clamped_primary,
            estimated_hold_usd=hold,
            rationale=(
                f"Selected primary open-weight model {primary.target} ({tier.value} tier, "
                f"max_tokens={clamped_primary}, estimated hold ${hold:.6f})."
            ),
        )
