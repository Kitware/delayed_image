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
