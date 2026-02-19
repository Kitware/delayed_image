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
