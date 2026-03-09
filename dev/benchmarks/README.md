Benchmark utilities for comparing `delayed_image` performance across refs.

Run a comparison against the default baseline and current `HEAD`:

```bash
python dev/benchmarks/compare_branches.py compare
```

Run an explicit comparison:

```bash
python dev/benchmarks/compare_branches.py compare --refs origin/main HEAD
```

Include the GDAL / COG-style patch sampling cases:

```bash
python dev/benchmarks/compare_branches.py compare --refs origin/main HEAD --include-gdal
```

Compare `origin/main` against the current working tree, including uncommitted changes:

```bash
python dev/benchmarks/compare_branches.py compare --refs origin/main WORKTREE --include-gdal
```

Outputs are written under `dev/benchmarks/results/compare-<timestamp>/`:

- `comparison.json`: raw measurements and metadata
- `*.json`: per-ref benchmark results
- `speedup_histogram.png`: speedup plot relative to the baseline ref

Notes:

- The benchmark env is shared across refs on purpose so dependency versions are
  held constant while the code changes.
- Ref checkouts are created as detached git worktrees under
  `dev/benchmarks/_cache/worktrees/`.
- GDAL cases are optional and install `requirements/gdal.txt` into a separate
  cached benchmark env.
