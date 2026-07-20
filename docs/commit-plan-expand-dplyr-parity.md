# Commit plan: `expand-dplyr-parity`

Goal: land the large uncommitted working tree as **reviewable, green commits**
without mixing CI scaffolding, implementation, oracle tests, and docs in one
blob.

Status at plan time: branch `expand-dplyr-parity`, last commit
`da817cb Expand dplyr parity and benchmark coverage`, plus ~3.9k lines of
modified/untracked work.

## Preconditions

```bash
pip install -e ".[dev]"
python -m pytest -q          # expect pass; R-oracle may skip without R
```

After each commit below, re-run `pytest -q` (and the performance suite only
when touching `bench_suite`).

Do **not** amend commits that are already on `origin` unless you intentionally
rewrite the remote branch.

## Recommended stack (8 commits)

### 1. CI: unit tests + performance workflow

**Purpose:** green checks before more surface area lands on the default branch.

```bash
git add \
  .github/workflows/tests.yml \
  .github/workflows/performance.yml
git commit -m "$(cat <<'EOF'
Add CI for unit tests and performance budgets.

Run pytest on Python 3.10–3.12 and keep an optional R oracle job for dplyr
semantic parity. Track the existing performance suite workflow so ratio
budgets fail PRs that regress adoption-critical paths.
EOF
)"
```

### 2. Expressions, tidyselect, and join specs

**Purpose:** backend-neutral expr surface and selectors without the large
verb/tidyr diffs.

```bash
git add \
  src/tidy3/expr.py \
  src/tidy3/tidyselect.py \
  src/tidy3/join_spec.py \
  tests/test_phase1.py \
  tests/test_phase2.py \
  tests/test_phase3.py
git commit -m "$(cat <<'EOF'
Expand backend-neutral expressions, tidyselect, and join_by.

Add ranking/window and coercion helpers, richer selectors, and join
condition coverage used by later verb and reshape work.
EOF
)"
```

If `tests/test_phase*.py` also assert verb behavior that is not yet committed,
either keep those assertions with commit 3 or split with `git add -p`.

### 3. Core verbs + frame + pandas engine

**Purpose:** dplyr verb bulk (grouping, rows_*, slices, set ops, etc.).

```bash
git add \
  src/tidy3/verbs.py \
  src/tidy3/frame.py \
  src/tidy3/pandas_engine.py \
  src/tidy3/__init__.py \
  tests/test_pandas_backend.py \
  tests/test_io_joins.py \
  tests/test_phase6.py \
  tests/test_phase8.py \
  tests/test_numpy_bridge.py
git commit -m "$(cat <<'EOF'
Expand dplyr verbs on Polars and pandas backends.

Extend frame methods, grouped/window semantics, row mutation, and join
paths. Keep dual-backend parity tests and NumPy handoff coverage in sync.
EOF
)"
```

Note: `__init__.py` may also export tidyr symbols — if so, either defer those
exports to commit 4 (`git add -p src/tidy3/__init__.py`) or include a no-op
import stub only when tidyr is present.

### 4. tidyr reshape and missing data

**Purpose:** isolate the new `tidyr` module and reshape tests.

```bash
git add \
  src/tidy3/tidyr.py \
  tests/test_reshape_missing.py
# plus any remaining tidyr exports in __init__.py
git add -p src/tidy3/__init__.py
git commit -m "$(cat <<'EOF'
Add tidyr-style reshape and missing-data verbs.

Implement pivot, separate/unite, nest/unnest, fill/complete/expand, and
related dual-backend coverage for missing values.
EOF
)"
```

### 5. R semantic oracle and dplyr parity contracts

**Purpose:** differential tests against real dplyr/tidyr (skip when R missing).

```bash
git add \
  tests/r_oracle_cases.R \
  tests/test_r_oracle_parity.py \
  tests/test_dplyr_semantic_parity.py
git commit -m "$(cat <<'EOF'
Add R oracle and dplyr semantic parity tests.

Compare public frame verbs to dplyr/tidyr when R is available, classify
every verb, and lock NA/group edge contracts on both backends.
EOF
)"
```

### 6. Benchmark suite updates

**Purpose:** adoption workloads and CI ratio budgets stay honest.

```bash
git add \
  src/tidy3/bench.py \
  src/tidy3/bench_suite.py \
  tests/test_bench_suite.py
git commit -m "$(cat <<'EOF'
Expand adoption benchmarks and per-op timing surfaces.

Cover multi-step workflows, handoff boundaries, and suite options used by
the performance CI budget.
EOF
)"
```

### 7. Docs: maturity tiers, development, package metadata

**Purpose:** set adoption expectations and point contributors at CI/tests.

```bash
git add \
  README.md \
  pyproject.toml \
  docs/commit-plan-expand-dplyr-parity.md
git commit -m "$(cat <<'EOF'
Document API maturity tiers and development workflow.

Label stable, growing, and experimental surfaces; describe pytest/R oracle
CI and how this branch is meant to land.
EOF
)"
```

### 8. (Optional) performance budget fix for experimental workloads

**Do not include until code exists.** If CI fails on geo-mean because
`group_nest` / group-callback workloads dominate, either:

- exclude those workloads from the primary geo budget, or
- mark them experimental in `bench_suite` and budget them separately

That change belongs in a follow-up PR (perf cliff work), not mixed into docs.

## One-shot alternative (not preferred)

If you need a single checkpoint and will rewrite history before merge:

```bash
git add -A
git commit -m "$(cat <<'EOF'
Checkpoint expand-dplyr-parity: verbs, tidyr, oracle, benches, CI.

Large WIP checkpoint. Prefer splitting via docs/commit-plan-expand-dplyr-parity.md
before merging to main.
EOF
)"
```

## After landing

1. Push the branch and confirm **Tests** + **Performance budget** on GitHub.
2. Open a PR into `main` (or stack smaller PRs matching commits 2–6).
3. Only then start the next phase (Polars-native `group_nest` / perf cliffs).

## File → commit map

| Path | Commit |
|------|--------|
| `.github/workflows/tests.yml` | 1 |
| `.github/workflows/performance.yml` | 1 |
| `src/tidy3/expr.py`, `tidyselect.py`, `join_spec.py` | 2 |
| `tests/test_phase1.py`, `test_phase2.py`, `test_phase3.py` | 2 |
| `src/tidy3/verbs.py`, `frame.py`, `pandas_engine.py` | 3 |
| `src/tidy3/__init__.py` | 3–4 (split exports if needed) |
| `tests/test_pandas_backend.py`, `test_io_joins.py`, `test_phase6.py`, `test_phase8.py`, `test_numpy_bridge.py` | 3 |
| `src/tidy3/tidyr.py`, `tests/test_reshape_missing.py` | 4 |
| `tests/r_oracle_cases.R`, `test_r_oracle_parity.py`, `test_dplyr_semantic_parity.py` | 5 |
| `src/tidy3/bench.py`, `bench_suite.py`, `tests/test_bench_suite.py` | 6 |
| `README.md`, `pyproject.toml`, `docs/commit-plan-expand-dplyr-parity.md` | 7 |
