# Mozilla Otari Integration & Engineering Feedback (`OTARI.md`)

**Project**: `Otari FinOps Guard & Adaptive Model Router`  
**Repository**: [`https://github.com/LBCron/otari-finops-guard`](https://github.com/LBCron/otari-finops-guard)  
**Hackathon**: Mozilla Otari Hackathon — Devs Wilding Online! (`devs-wilding` & `main` branches)  
**Author**: `@LBCron` (`LBCron <ronanchenlopes@gmail.com>`)

---

## 1. How Mozilla Otari Is Integrated (All 8 Rubric Capabilities + 4 Open-Weight Models)

`Otari FinOps Guard & Adaptive Model Router` is a FinOps control plane, adaptive open-weight model router, and deterministic checkpointing harness purpose-built around Mozilla Otari's gateway primitives.

### A. Open-Weight Models Serving 4 Distinct Roles (`src/otari_finops_guard/router.py`)
Rather than routing every subagent call to a single frontier model, our orchestrator maps four specialized software-engineering subagent roles onto **four distinct open-weight models** (with automatic economy open-weight fallbacks above the 75% budget watermark):

| Subagent Role | Primary Open-Weight Model | Economy Open-Weight Fallback (`>=75%` Budget) | Why This Model Fits the Role |
| :--- | :--- | :--- | :--- |
| **`architect`** | `deepseek:deepseek-r1` | `together:meta-llama/Llama-3.3-70B-Instruct` | DeepSeek-R1 (671B MoE reasoning model) excels at multi-file root-cause localization and architectural decomposition. |
| **`researcher`** | `together:meta-llama/Llama-3.3-70B-Instruct` | `mistral:mistral-small-24b-instruct-2501` | Llama-3.3-70B-Instruct synthesizes multi-document web search and markdown fetch results with high factual fidelity. |
| **`coder`** | `together:Qwen/Qwen2.5-Coder-32B-Instruct` | `mistral:mistral-small-24b-instruct-2501` | Qwen2.5-Coder-32B-Instruct matches frontier models on surgical Python diff generation and unit-test synthesis at a fraction of the cost. |
| **`verifier`** | `mistral:mistral-small-24b-instruct-2501` | `mistral:mistral-small-24b-instruct-2501` (clamped tokens) | Mistral-Small-24B-Instruct-2501 delivers low-latency diff verification and sandbox test output auditing at `$0.00007 / 1K` input tokens. |

### B. All 8 Mozilla Otari Gateway Capabilities Used in Application Code

1. **Model Routing & Fallbacks (`src/otari_finops_guard/router.py`, `src/otari_finops_guard/client.py`)**:
   - Queries `POST /api/v1/routing/recommend` (`harness: "pi"`) with role-annotated open-weight model candidates.
   - Implements a **4-state Budget Watermark Router** (`OPTIMAL` `<50%`, `RIGHT_SIZED` `50–75%`, `DEGRADED_OPEN_WEIGHT` `75–99%`, `HALTED_CHECKPOINTED` `100%`) that automatically downgrades `together:Qwen/Qwen2.5-Coder-32B-Instruct` to `mistral:mistral-small-24b-instruct-2501` once a subagent crosses 75% of its Otari budget.
   - Tracks `Otari-Provider`, `Otari-Model`, `Otari-Attempt-Count`, and `Otari-Fallback` response headers.
2. **Per-User / Per-Subagent Budgets (`src/otari_finops_guard/client.py`, `src/otari_finops_guard/swarm.py`)**:
   - Uses Otari **Service Keys** to isolate each subagent in a multi-agent coding swarm under its own end-user budget (`GET /api/v1/end-user-budgets`, `PUT /api/v1/end-users?external_id=subagent:coder`, `GET /api/v1/end-users?external_id=...`).
   - Every inference call passes `"user": "subagent:<role>"` and `Otari-End-User-Budget: subagent-<role>-budget`, reading back the confirmed `Otari-End-User-Budget` header.
3. **Budgets, Spending Limits & Upfront Hold Right-Sizing (`src/otari_finops_guard/router.py`)**:
   - Inspects key-wide limits via `GET /api/v1/key/budget` and tag-grouped spend via `GET /api/v1/key/usage/summary?group_by_tag=role&dimensions=none` (paired with `Otari-Tags: role=<role>,task=<task_id>`).
   - **Solves Otari's Upfront Reservation Cliff**: Because Otari's `reserve_budget` holds `input_cost + (max_tokens * output_rate)` upfront before calling the provider, a request with a default `max_tokens=2048` will be refused with `403 budget_exceeded` when a subagent has `$0.00035` left—even if the actual response only needs 300 tokens (`$0.00009`). `AdaptiveFinOpsRouter.clamp_max_tokens_to_fit_hold()` dynamically computes the exact maximum `max_tokens` whose worst-case hold fits inside `remaining_usd`.
4. **MCP Integrations (`src/otari_finops_guard/client.py`, `src/otari_finops_guard/swarm.py`)**:
   - Discovers gateway-registered MCP servers via `GET /api/v1/mcp/servers`, invokes tools directly via `POST /api/v1/mcp/execute` (`repo-inspector-mcp:inspect_otari_issue`), and attaches `mcp_servers: ["repo-inspector-mcp"]` to chat completions so gateway-side MCP calls are recorded in `otari_tool_calls`.
5. **Web Search & Web Fetch (`src/otari_finops_guard/client.py`, `src/otari_finops_guard/swarm.py`)**:
   - Invokes `POST /api/v1/search` and `POST /api/v1/fetch` directly and enables gateway-executed search/fetch during inference via `Otari-Web-Search: otari` and `Otari-Web-Fetch: otari` headers.
6. **Sandboxed Code Execution (`src/otari_finops_guard/client.py`, `src/otari_finops_guard/swarm.py`)**:
   - Executes unit test verification snippets via `POST /api/v1/code/execute` and enables model-driven sandbox execution via `Otari-Code-Execution: otari`.
7. **Guardrails (`src/otari_finops_guard/client.py`, `src/otari_finops_guard/swarm.py`)**:
   - Attaches per-request `guardrails: [{"profile": "prompt-injection", "mode": "block", "on": "input"}]` to block untrusted issue comments attempting prompt injection or secret exfiltration (`OTARI_MASTER_KEY`), and parses the `Otari-Guardrails` telemetry header.
8. **Idempotency Replay & Deterministic 100% Checkpointing (`src/otari_finops_guard/swarm.py`)**:
   - Sends `Idempotency-Key` on side-effecting steps and verifies `Otari-Idempotent-Replayed: true` with `$0.000000` duplicate charge on network retries.
   - At 100% budget exhaustion, cleanly halts the subagent and serializes a verifiable `SwarmTaskCheckpoint` JSON artifact (`artifacts/task-otari-issue-2992.checkpoint.json`).

---

## 2. Written Feedback Answers (DoraHacks Submission Questions)

### Q1. How was your overall experience using Otari? What worked well and what felt smooth?
- **Unified FinOps + Tooling Surface**: Having OpenAI (`/api/v1/chat/completions`), Anthropic (`/api/v1/messages`), self-inspection (`/api/v1/key/budget`, `/api/v1/key/usage/summary`), service-key per-user budgets (`/api/v1/end-users`), built-in tools (`Otari-Web-Search`, `Otari-Web-Fetch`, `Otari-Code-Execution`), and MCP execution (`/api/v1/mcp/execute`) behind a single gateway eliminated glue code across four different model providers (`deepseek`, `together`, `mistral`, `anthropic`).
- **Actionable Budget Refusal Diagnostics**: When an upfront hold exceeds remaining headroom, Otari's `refusal_detail()` in `src/gateway/services/budgets/_refusals.py` explicitly tells the caller both dollar amounts AND the assumed output token allowance:
  > `"... has $0.003747 of budget remaining, less than this request's estimated cost of $0.008224. The estimate assumes 8192 output tokens; a lower output token limit on the request may let it fit"`
  This design directly inspired our `clamp_max_tokens_to_fit_hold()` algorithm in `AdaptiveFinOpsRouter`.
- **Rich Telemetry Headers**: `Otari-Request-ID`, `Otari-Provider`, `Otari-Model`, `Otari-Attempt-Count`, `Otari-Fallback`, `Otari-End-User-Budget`, `Otari-Guardrails`, `Otari-Idempotent-Replayed`, and `Server-Timing` make building real-time observability dashboards effortless without scraping database logs.

### Q2. Where did you get stuck, confused, or frustrated?
1. **Service-Key End-User Budget Refusals Leak Internal `eu_<hex>` IDs (Issue [`mozilla-ai/otari#2992`](https://github.com/mozilla-ai/otari/issues/2992))**:
   - When using a Service Key to cap multiple subagents via `"user": "subagent:coder"`, Otari creates a `User` row with an internal `user_id = "eu_<32-hex>"` and stores `"subagent:coder"` in `User.external_id`.
   - However, when that end user hits its budget cap or is blocked, `reserve_budget` in `src/gateway/services/budgets/_reservations.py` (lines 800 and 1035) formats `f"User '{user_id}'"` (`User 'eu_a886c78298c7419caf4764794945248e'`) instead of `End user 'subagent:coder'`. A multi-agent orchestrator cannot tell from the 403 message which of its subagents or customers was refused!
2. **Guardrail 403 Responses Omit `Otari-Error-Code` Header and Nest `code` Inside `detail` (Issue [`mozilla-ai/otari#2989`](https://github.com/mozilla-ai/otari/issues/2989))**:
   - Every other gateway refusal sets `Otari-Error-Code: <code_name>` and puts `"code"` at the top level of the JSON body.
   - In `src/gateway/api/routes/_helpers.py` (lines 267–282), `guardrail_violation` raises `HTTPException(status_code=403, detail={"error": ..., "code": "guardrail_violation"})` without `headers=error_headers(GUARDRAIL_VIOLATION)`. Clients keying on `Otari-Error-Code` miss guardrail blocks unless they also inspect `body["detail"]["code"]`.
3. **Named Routing Policies Do Not Fall Back When Candidate #1 Hits a Provider-Narrowed Spend Ceiling (Issue [`mozilla-ai/otari#2981`](https://github.com/mozilla-ai/otari/issues/2981))**:
   - In `src/gateway/api/routes/_pipeline.py` (line 3212), the upfront `reserve_budget` loop only skips to the next candidate on `BUDGET_EXCEEDED` when `plan.automatic is not None` (`model="auto"`), refusing named routing policies (`plan.policy is not None`) even when the second candidate uses a different provider whose spend ceiling still has room.

### Q3. Were there any features or capabilities you wished existed but didn't?
1. **Gateway-Native `Otari-Clamp-Max-Tokens: true` Header**:
   - Right now, if a request asks for `max_tokens=4096` (`$0.0082` hold) and the user has `$0.0037` remaining, Otari refuses with 403 and suggests lowering `max_tokens`. An opt-in request header (`Otari-Clamp-Max-Tokens: min=128`) that lets Otari automatically clamp the request's `max_tokens` to fit the remaining budget hold would prevent round-trip 403 retries for autonomous agents.
2. **End-User Identifier in Budget Refusal JSON Body (`"end_user": "<external_id>"`)**:
   - Alongside naming `End user '<external_id>'` in `detail` (which our patch in `patches/otari-issue-2992.patch` fixes), including `"end_user": "<external_id>"` as a machine-readable field in 403 `budget_exceeded` responses would make multi-tenant routing even cleaner.
3. **Webhook / SSE Budget Watermark Events (`75%` Warning, `100%` Exhausted)**:
   - Emitting an `Otari-Budget-Utilization-Pct` response header on every completion would let clients adjust model routing tiers without polling `GET /api/v1/key/budget` or `GET /api/v1/end-users`.

---

## 3. Verified Upstream Fix Patch Contributed (`patches/otari-issue-2992.patch`)

We traced, fixed, and unit-tested **Issue [`mozilla-ai/otari#2992`](https://github.com/mozilla-ai/otari/issues/2992)** directly against the `mozilla-ai/otari` codebase (included in [`patches/otari-issue-2992.patch`](patches/otari-issue-2992.patch) and [`patches/otari-issue-2992-PR.md`](patches/otari-issue-2992-PR.md)):

- **Root Cause**: `reserve_budget` in `src/gateway/services/budgets/_reservations.py` (lines 800 and 1035) passed `f"User '{user_id}'"` to `refusal_detail(...)` even when `user.external_id is not None`.
- **Surgical Fix**:
  1. Added `user_subject(user_id: str, *, external_id: str | None = None) -> str` in `src/gateway/services/budgets/_refusals.py` returning `f"End user '{external_id}'"` when `external_id is not None` and `f"User '{user_id}'"` otherwise (matching `EndUserNotFoundError`).
  2. Updated `reserve_budget` in `src/gateway/services/budgets/_reservations.py` to use `user_subject(...)` for both `USER_BLOCKED` and `BUDGET_EXCEEDED` 403 refusals.
  3. Verified with `pytest tests/unit/test_budget_refusal_detail.py` (`8 passed`) and `ruff check` (`All checks passed!`).
