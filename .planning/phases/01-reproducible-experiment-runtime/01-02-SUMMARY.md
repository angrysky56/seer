---
phase: 01-reproducible-experiment-runtime
plan: "02"
subsystem: experiment-runtime
tags: [python, transactions, provenance, checkpointing, reproducibility]
requires: [01-01]
provides:
  - Atomic artifact writes with SHA-256 inventory and fail-closed finalization
  - Complete run provenance manifests and immutable completion markers
  - Explicit locking, replacement, compatible resume, and deterministic RNG checkpoints
affects: [01-04, experiment-arms, reporting]
tech-stack:
  added: []
  patterns: [temporary sibling replacement, content-addressed run identity, explicit resume position]
key-files:
  created: [src/seer/runtime.py, tests/test_runtime.py]
  modified: []
key-decisions:
  - Completion is a final commit marker written only after artifact revalidation.
  - Replacements preserve the previous run directory and record the event in new state.
  - Checkpoints persist next epoch and batch plus sampler state rather than deriving position from steps.
patterns-established:
  - Mutable state, locks, temporary files, and manifests are excluded from artifact inventory.
  - Stale locks require explicit recovery and preserve the prior lock as audit evidence.
requirements-completed: [EXP-03, EXP-04]
duration: 16 min
completed: 2026-07-16
---

# Phase 1 Plan 02: Transactional Runtime Summary

**Fail-closed artifact provenance, immutable config-addressed runs, auditable locks, and exact checkpoint-boundary RNG restoration**

## Performance

- **Duration:** 16 min
- **Completed:** 2026-07-16T05:15:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added durable sibling-temporary writes, relative artifact inventory, size/SHA-256 validation, comprehensive environment provenance, and completion-time revalidation.
- Added config-addressed run creation with explicit completed no-op, compatible incomplete resume, preserved replacement directories, and auditable recovery events.
- Added checkpoint contracts for model/optimizer progress, schema/config identity, Python/NumPy/Torch CPU and CUDA RNG, named generators, explicit next epoch/batch, and sampler state.

## Task Commits

Each test-first task was committed atomically:

1. **Task 01-02-01: Implement atomic artifact and manifest provenance primitives**
   - `c1d4165` (test)
   - `ca14a2f` (feat)
2. **Task 01-02-02: Enforce run lifecycle, locking, and deterministic checkpoint contracts**
   - `40015c8` (test)
   - `69fc84a` (feat)

## Files Created/Modified

- `src/seer/runtime.py` - Atomic I/O, artifact validation, provenance manifests, run state, locking, replacement/resume, RNG, and checkpoints.
- `tests/test_runtime.py` - Failure injection, tamper/deletion, provenance, lock/recovery, lifecycle, replacement/resume, and RNG round-trip tests.

## Decisions Made

- Used canonical configuration digest plus a readable run-name slug for stable logical identity.
- Made ordinary reruns of completed compatible runs successful no-ops while rejecting resume against a completion marker.
- Required callers to supply named generators during restoration so missing deterministic data-order state fails explicitly.

## Deviations from Plan

None - plan scope and behavior were implemented as written.

## Issues Encountered

- Focused Plan 01-02 tests and lint passed. The full 73-test suite also passed. Full-repository Ruff remains blocked by four pre-existing style findings in `src/seer/eval.py`, `tests/test_optim.py`, and `tests/test_train.py`; these files are outside Plan 01-02 ownership and were left untouched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 01-04 can use `RunStore`, artifact inventories, manifests, and checkpoints for its synthetic vertical slice.
- Later experiment arms can supply model/dataset identifiers and declared artifacts without changing transactional semantics.

## Self-Check: PASSED

- Key files exist and all four `01-02` test/feature commits are present.
- `uv run pytest tests/test_runtime.py -q`: 7 passed.
- `uv run ruff check src/seer/runtime.py tests/test_runtime.py`: passed.
- `uv run pytest`: 73 passed.
- Task acceptance criteria were exercised by failure-case tests and passed.

---
*Phase: 01-reproducible-experiment-runtime*
*Completed: 2026-07-16*
