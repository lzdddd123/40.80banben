# Session Splitting Design

## Goal
Add an optional, low-risk internal session-splitting enhancement on top of the current `40.70` baseline without changing the graph encoder, graph construction, or core scoring path for sessions that do not trigger splitting.

The feature targets only long sessions, where multiple intents are more likely to coexist. The design keeps the original full-session prediction as the dominant path and blends in split-session predictions with a small weight.

## Scope
In scope:
- Runtime-configurable splitting threshold
- Runtime-configurable split-fusion weight
- Fixed-ratio hard splitting for long sessions only
- Reuse of the existing `compute_scores()` path for full/front/back views

Out of scope:
- Dynamic split boundary learning
- New encoders, new graph views, or new CL variants
- Changes to data preprocessing or graph construction
- Dataset-specific branching for split behavior

## Recommended Approach
Use fixed hard splitting with a shared rule across datasets:
- If valid session length `<= split_threshold`, use the original full-session scoring only.
- If valid session length `> split_threshold`, split the valid portion into:
  - front segment: first `60%`
  - back segment: remaining `40%`
- Compute scores for:
  - the full session
  - the front segment
  - the back segment
- Blend them as:

`final_scores = (1 - split_lambda) * full_scores + split_lambda * 0.5 * (front_scores + back_scores)`

This preserves the original baseline path while allowing long sessions to benefit from reduced intent mixing.

## Why This Approach
This is the most conservative design that still tests the core hypothesis.

Reasons:
- It does not alter the graph encoder.
- It does not change the data or baseline comparison protocol.
- It does not replace the current `max-over-interests` inference path.
- It is easy to disable completely by setting a large threshold or a zero lambda.
- It can be evaluated cleanly as a lightweight model-side enhancement.

## Component Changes

### `main.py`
Add two arguments:
- `--split_threshold`
  - default: `999`
  - meaning: disabled for all practical session lengths
- `--split_lambda`
  - default: `0.1`
  - meaning: small fusion weight to reduce risk to the original HR advantage

No other training logic changes are planned.

### `model.py`
Refactor scoring so that the existing score computation can be reused with alternate masks:
- Keep the existing `compute_scores(hidden, mask)` behavior as the base path
- Add an internal helper that computes scores for a supplied mask
- Add optional split logic:
  - determine valid lengths from `mask`
  - generate front/back masks only for samples with `length > split_threshold`
  - compute front/back scores by reusing the same scoring helper
  - blend scores only for the triggered samples

### Files That Must Not Change
- `aggregator.py`
- `build_graph.py`
- `utils.py`

This is intentional to preserve the current baseline behavior as much as possible.

## Data Flow
1. The encoder produces `seq_hidden` exactly as before.
2. The full-session score is computed exactly as before.
3. If a sample is longer than `split_threshold`:
   - build a front mask over the first `60%` valid positions
   - build a back mask over the remaining valid positions
   - compute front and back scores using the same scoring code
   - blend them into the final score
4. If a sample does not trigger splitting, the final score equals the full-session score.

## Boundary Rules
To avoid degenerate segments:
- splitting applies only when `valid_len > split_threshold`
- both front and back segments must contain at least one valid position
- if a split would create an empty segment, fall back to the original full-session score

Initial recommended runtime settings:
- `split_threshold = 15`
- `split_lambda = 0.1`

## Error Handling
- If `split_threshold` is too small and causes unstable behavior, users can disable the feature by raising the threshold.
- If `split_lambda = 0`, behavior should match the original baseline.
- If a segment mask becomes invalid, the implementation must fall back to the original full-session score for that sample.

## Evaluation Plan
Start with four comparisons:
- `Tmall` baseline
- `Tmall` with split enabled
- `retailrocket` baseline
- `retailrocket` with split enabled

Primary focus:
- preserve HR on `Tmall`
- improve long-session behavior on `retailrocket`

## Testing Plan
- Syntax validation for `main.py` and `model.py`
- Existing tests should continue to pass
- Add a small unit test for split-mask generation if the implementation is isolated enough
- Run a smoke test with splitting disabled to confirm parity
- Run a smoke test with `split_threshold=15`, `split_lambda=0.1`

## Success Criteria
- No regression when splitting is effectively disabled
- Long-session-only enhancement can be toggled from the command line
- The baseline path remains intact for short and medium sessions
- The feature is simple enough to explain as a controlled extension rather than a new architecture
