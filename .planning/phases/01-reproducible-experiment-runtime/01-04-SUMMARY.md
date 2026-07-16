---
phase: 01-reproducible-experiment-runtime
plan: "04"
subsystem: experiment-runtime
tags: [determinism, checkpointing, smoke, offline, operations]
requires: [01-01, 01-02, 01-03]
provides:
  - Deterministic checkpoint-aware CPU training continuation
  - Offline synthetic CLI experiment using production artifact contracts
  - Scientific equivalence, immutable no-op, and tamper gates
  - Complete local runtime and cached-model operations guide
affects: [phase-02, phase-03, phase-04, phase-05]
tech-stack:
  added: []
  patterns: [explicit deterministic permutation, next-position cursor, canonical scientific records]
key-files:
  created: [src/seer/smoke.py, tests/test_smoke.py, docs/OPERATIONS.md]
  modified: [src/seer/train.py, src/seer/cli.py, tests/test_train.py, README.md]
key-decisions:
  - Scientific equivalence hashes include only canonical step records, excluding operational metadata.
  - Synthetic smoke owns a package-local fake model and never crosses the Transformers boundary.
  - Completed-run execution is a verifying no-op and fails closed when any inventoried artifact changes.
patterns-established:
  - TrainCursor identifies the next logical epoch and batch, preventing duplicate resumed steps.
  - Checkpoint callbacks carry model, optimizer, RNG, data-order, cursor, and scientific-record state.
requirements-completed: [EXP-05, DOC-01, QUAL-01, QUAL-02, QUAL-04]
duration: 24 min
completed: 2026-07-16
---

# Phase 1 Plan 04: Deterministic Offline Smoke Runtime Summary

**A resumable CPU synthetic experiment now exercises SEER's real config, checkpoint, result,
manifest, integrity, and completion contracts without external weights, data, or network access.**

## Performance

- **Duration:** 24 min
- **Completed:** 2026-07-16
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Refactored training around an explicit next-position cursor, seeded epoch permutations, stable
  scientific step records, and callbacks containing every continuation input.
- Added a package-owned tiny model and `seer smoke` handler that checkpoints during work, resumes
  interrupted evidence, writes JSON and JSONL results, inventories artifacts, and commits
  `COMPLETE` last.
- Proved clean and resumed scientific hashes are identical, completed reruns change no bytes, and
  corrupt completed artifacts produce a nonzero diagnostic.
- Documented setup, offline gates, smoke, resume, replacement, deliberate lock recovery, artifact
  semantics, pinned cache preflight, explicit download, and the manual-only real forward check.

## Task Commits

1. **Task 01-04-01: Add deterministic checkpoint-aware training continuation**
   - `81e1e76` (feat)
2. **Task 01-04-02: Wire the synthetic CLI vertical slice and equivalence gate**
   - `d512461` (feat)
3. **Task 01-04-03: Document operations and close the offline quality gate**
   - `6f5e0ee` (docs)

## Files Created/Modified

- `src/seer/train.py` - Deterministic seeding, cursor, scientific records, and checkpoint callback.
- `src/seer/smoke.py` - Package-owned fake model and transactional synthetic experiment.
- `src/seer/cli.py` - Lazy built-in smoke dispatch with actionable failure diagnostics.
- `tests/test_train.py` - Deterministic order, cadence, cursor, and continuation equivalence tests.
- `tests/test_smoke.py` - Offline tree, no-op, interruption/resume, and tamper integration gates.
- `README.md` - Copyable local smoke entry point.
- `docs/OPERATIONS.md` - Full Phase 1 operator guide and safety boundaries.
- `src/seer/eval.py`, `tests/test_optim.py` - Small repository-wide Ruff cleanup.

## Decisions Made

- Epoch order is derived from `training_seed + epoch`, while checkpoints retain the current
  permutation and explicit next position for compatibility validation and auditability.
- The result envelope stores a canonical hash over scientific records only. Manifest timestamps,
  hardware facts, and interruption history cannot change the scientific equivalence result.
- The public CLI has no test interruption flag; interruption injection remains an internal Python
  seam, while ordinary process termination resumes from the latest durable checkpoint.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Closed three pre-existing repository-wide Ruff findings**
- **Found during:** Task 01-04-03 full quality gate
- **Issue:** One long docstring line and two import-order findings prevented QUAL-02 despite being
  outside the task's primary files.
- **Fix:** Wrapped the docstring and applied Ruff's import ordering to the two affected tests.
- **Files modified:** `src/seer/eval.py`, `tests/test_optim.py`, `tests/test_train.py`
- **Verification:** `uv run ruff check .` passes and the full suite remains green.
- **Committed in:** `6f5e0ee`

---

**Total deviations:** 1 auto-fixed (Rule 1)
**Impact on plan:** Required to satisfy the plan-owned repository-wide lint gate; no scope expansion.

## Issues Encountered

- The sandbox cannot write uv's default cache. All commands used `UV_CACHE_DIR=/tmp/seer-uv-cache`
  against the existing locked environment; no dependencies, model weights, or datasets downloaded.
- Torch emitted an environment-only NVML initialization warning; CPU execution and all assertions
  completed successfully.

## Verification

- `uv run pytest tests/test_train.py -q` - **6 passed**.
- Focused config/runtime/CLI/smoke/train gate - **31 passed**.
- `uv run pytest` - **78 passed** in 3.28 seconds.
- `uv run ruff check .` - **all checks passed**.
- `uv lock --check` - **lock is current**.
- `uv run seer --help` - **all milestone commands listed**.
- Documented smoke command run twice in `/tmp/seer-plan04-smoke.cvpi4U` - **all file hashes
  unchanged**, artifact tree matched documentation.
- Interrupted/resumed equivalence and completed-evidence tamper rejection - **passed in integration
  tests**.
- Cached Qwen preflight - **passed earlier in Plan 01-03 execution context**; real forward remained
  manual-only as required.

## User Setup Required

None. Downloads and real-model loading remain explicit operator actions.

## Next Phase Readiness

- Phase 1's reproducible runtime is complete and ready for Phase 2 normalized real-domain adapters.
- Later experiment stages can reuse the cursor/checkpoint seam and versioned result envelope without
  making synthetic tests depend on external resources.

## Self-Check: PASSED

---
*Phase: 01-reproducible-experiment-runtime*
*Completed: 2026-07-16*
