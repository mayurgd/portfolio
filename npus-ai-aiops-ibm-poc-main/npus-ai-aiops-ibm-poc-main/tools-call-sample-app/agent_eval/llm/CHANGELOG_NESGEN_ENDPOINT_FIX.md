# NESGEN Endpoint Request Fix Notes

## Summary
This document records the recent changes made to `nestle_chat_model_v2.py` to resolve runtime HTTP errors in Streamlit while using NESGEN.

## What changed

### 1) Added endpoint-aware payload selection for `NESGEN_URL`
When `NESGEN_URL` is set, `_make_request` now chooses payload shape based on URL path:

- If URL contains `/completions`:
  - sends chat-completions style payload:
  - `{"messages": [...], "temperature": ...}`

- Otherwise (for responses-style endpoints):
  - sends responses style payload:
  - `{"model": ..., "input": ...}`

Reason:
- The full NESGEN URL in use was a `/completions` endpoint.
- Sending responses-style payload to `/completions` caused HTTP 400 errors.
- This preserves compatibility with both endpoint styles without changing higher-level app logic.

## Root-cause findings (during debugging)
1. Initial 400 errors were caused by payload/endpoint mismatch:
   - endpoint: `/completions`
   - payload sent: responses style (`model` + `input`)

2. A later NOT_FOUND error came from a duplicated path in fallback mode:
   - `.../openai/deployments/openai/deployments/...`
   - This was due to using an `NESGEN_API_BASE` value that already included `openai/deployments`.

3. A subsequent 404 indicated model/action/version resolution mismatch in fallback mode.
   Using the exact provided full `NESGEN_URL` with the correct payload shape resolved the primary issue.

## Operational note
After changing `.env` values (`NESGEN_URL`, `NESGEN_API_BASE`, `NESGEN_MODEL`, etc.), restart Streamlit so environment variables are reloaded.
