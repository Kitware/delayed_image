# Codex Journal

## 2026-02-19 — Commit in progress

I focused on the recurring `test_off_by_one_with_small_img` failure that only appears in sdist CI while most local and strict-focused environments pass. My current belief is that runtime differences around `kwimage.warp_affine` matrix conventions and float border handling can still leak through despite probing.

What I changed in this step:
- Strengthened nearest-mode candidate selection to score both matrix-convention outputs against source diversity, not just finite coverage.
- Added richer debug logging (behind `DELAYED_IMAGE_WARP_DEBUG`) to print source unique count, both scores, and value previews for each candidate.

What I was thinking:
- If one candidate is the pathological `[0.8, nan]` output and the other preserves source values, we should deterministically choose the latter.
- If both candidates look bad in some stack, the new diagnostics should reveal whether this is a matrix issue, border-value issue, or backend dispatch issue.

Where this might go next:
- If CI still fails, the logs should tell us whether to add a backend pin/override for nearest or a more explicit border normalization for float64.
- Could add a tiny targeted regression test around the candidate scoring routine if needed.

## 2026-02-19 — Commit in progress (diagnostics expansion)

I was asked to add more diagnostic output in the failing test and keep pushing on fixes. I decided to improve diagnostics in two places at once: the test assertion message and the warp runtime path.

What I changed in this step:
- Enhanced `tests/test_off_by_one.py::test_off_by_one_with_small_img` to avoid opaque broadcast `ValueError` and instead report shapes, sample unique values, and finite ratio in the assertion message.
- Added a nearest/floating border-value override in `DelayedWarp._finalize()` that prefers scalar `np.nan` border values over `(np.nan,)` for warp calls, based on prior observations that this can differ by runtime stack.

What I was thinking:
- Better failure messages reduce guesswork and should immediately show whether this is a uniqueness-collapse issue, NaN-coverage issue, or something else.
- The scalar-vs-tuple NaN border handling has shown stack-dependent behavior before, so this is a low-risk compatibility lever worth trying.

Where this might go next:
- If CI still fails, I want to log both candidate outputs in the exact failing environment and compare not only uniqueness but also whether source values are preserved as a set.
- If needed, we can add a narrowly scoped nearest-upscale fallback path specialized for pure scale transforms.

## 2026-02-19 — Commit in progress (deeper hypothesis)

I think there is a deeper issue than just matrix-direction probing: in one sdist runtime, both matrix candidates may degrade in nearest mode for pure scale, which suggests a backend/pathology around warp + border interactions.

What I changed in this step:
- Added richer diagnostics to the failing test that explicitly compute direct forward/inverse `kwimage.warp_affine` baselines and include their finite ratios / unique previews in the assertion message.
- Added a narrowly scoped fallback in `DelayedWarp._finalize()` for nearest pure-scale transforms: if both candidate warp scores are pathologically low in finite coverage, rescue via `kwimage.imresize(..., interpolation='nearest')`.

What I was thinking:
- This keeps behavior stable for normal cases while giving us a deterministic escape hatch for the exact pathological signature in CI.
- The extra test diagnostics should show if the environment is failing both affine directions or only one.

Where this might go next:
- If this still fails, the next likely step is explicitly pinning nearest pure-scale to a backend-specific implementation or introducing a dedicated helper with direct OpenCV `resize` for that niche path.

## 2026-02-19 — Commit in progress (intermediate-state hypothesis)

I noticed one important issue in the diagnostics: the direct forward/inverse baseline in the test was accidentally using the *later* warp variable (scale+translation), not the `data1` warp (pure 8.6/8.5 scale). That can mislead analysis.

What I changed in this step:
- Fixed test diagnostics to use an explicit `data1_warp` for direct forward/inverse baseline comparisons.
- Added a deterministic nearest pure-scale fast-path in `DelayedWarp._finalize()` that immediately uses `kwimage.imresize(..., interpolation='nearest')` for near-zero-offset, no-rotation/shear, positive-scale transforms.
- Kept the dual-candidate affine scoring path as fallback for non-pure-scale nearest cases.

What I was thinking:
- This aligns behavior with the semantics expected by the failing assertion (nearest upscale should preserve source unique values).
- Pure-scale nearest is the exact case of the failing `data1`, so a direct resize path should remove stack-sensitive affine convention ambiguity.

Where this might go next:
- If CI still fails, we should log transform decomposition and `is_near_scale_only` status directly in assertion diagnostics to verify the fast-path is actually being hit.

## 2026-02-19 — Commit in progress (deeper intermediate inspection)

I found another subtle diagnostics issue: the baseline forward/inverse checks were using `x.dsize` after `x` had been reassigned for `data2`, so the baseline canvas size could be wrong for `data1` comparisons.

What I changed in this step:
- Introduced a deterministic axis-aligned nearest path in `DelayedWarp._finalize()` that performs manual corner-based nearest sampling (using floor on transformed coordinates) with explicit border fill.
- Fixed `test_off_by_one_with_small_img` diagnostics to capture `data1_dsize` immediately after computing `data1` and use that size for baseline forward/inverse checks.

What I was thinking:
- A manual nearest sampler avoids backend quirks and matrix-convention ambiguity for the exact class of transforms we care about.
- Correct diagnostic canvas size is essential; otherwise we may draw conclusions from mismatched outputs.

Where this might go next:
- If this finally stabilizes CI, we should consider extracting the deterministic nearest sampler into a helper and adding focused unit tests around it.
