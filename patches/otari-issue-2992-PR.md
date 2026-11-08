# fix(budgets): name service-key end users by external_id in budget refusals

Closes #2992

## Summary
When a request billed through a service key (`api_key.is_service_key`) names an end user in `user` (or `metadata.user_id`), Otari resolves or creates a `User` row whose internal `user_id` is `eu_<hex>` and whose caller-supplied identifier is stored in `User.external_id`.

Previously, `reserve_budget` in `src/gateway/services/budgets/_reservations.py` formatted both `user_blocked` and `budget_exceeded` 403 refusal details with `f"User '{user_id}'"`, leaking the opaque internal `eu_<hex>` ID that a service-key caller never sees instead of the `external_id` the request named.

## Changes
- Added `user_subject(user_id, *, external_id=None)` in `src/gateway/services/budgets/_refusals.py` to format `f"End user '{external_id}'"` when `external_id` is present and `f"User '{user_id}'"` otherwise (matching `EndUserNotFoundError` and `EndUserPathInvalidError` wording).
- Updated `reserve_budget` in `src/gateway/services/budgets/_reservations.py` to use `user_subject(...)` for both `USER_BLOCKED` and `BUDGET_EXCEEDED` 403 refusals.
- Added unit tests in `tests/unit/test_budget_refusal_detail.py` covering `user_subject` and `reserve_budget` refusal messages for service-key end users.

## Verification
- `pytest tests/unit/test_budget_refusal_detail.py -q` (`8 passed`)
- `ruff check src/gateway/services/budgets/_refusals.py src/gateway/services/budgets/_reservations.py tests/unit/test_budget_refusal_detail.py` (`All checks passed!`)
