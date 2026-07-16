---
phase: 01-reproducible-experiment-runtime
plan: "01"
subsystem: experiment-runtime
tags: [python, dataclasses, json, argparse, reproducibility]
requires: []
provides:
  - Strict versioned experiment configuration wire contract
  - Stable canonical configuration digest
  - Complete installed CLI command surface with injectable dispatch
affects: [01-02, 01-03, 01-04]
tech-stack:
  added: []
  patterns: [strict recursive decoding, dotted diagnostics, late-bound CLI handlers]
key-files:
  created: [src/seer/cli.py, tests/test_config.py, tests/test_cli.py, examples/synthetic.json]
  modified: [src/seer/config.py, pyproject.toml]
key-decisions:
  - Standard-library dataclasses remain the single configuration source of truth.
  - JSON is normalized at the schema boundary; runtime-derived facts stay outside the digest.
  - CLI parsing and synthetic dispatch do not import model or dataset implementations.
patterns-established:
  - Unknown nested configuration keys fail with dotted field paths.
  - Future commands are visible but return an honest nonzero not-yet-implemented result.
requirements-completed: [EXP-01, EXP-02]
duration: 18 min
completed: 2026-07-16
---

# Phase 1 Plan 01: Configuration and CLI Summary

**Strict JSON configuration with stable SHA-256 identity and an offline-safe, injectable seven-command CLI**

## Performance

- **Duration:** 18 min
- **Completed:** 2026-07-16T04:40:32Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added strict recursive decoding, validation, human serialization, canonical serialization, and stable digesting for every Phase 1 configuration group.
- Added the complete milestone CLI surface with auditable runtime overrides and mutually exclusive resume/replace policies.
- Added a fully explicit synthetic JSON fixture and parser seams that remain independent of Hugging Face model construction.

## Task Commits

Each test-first task was committed atomically:

1. **Task 01-01-01: Define the versioned configuration wire contract**
   - `04fd36e` (test)
   - `0c78aad` (feat)
2. **Task 01-01-02: Add the auditable CLI command surface**
   - `3e9de65` (test)
   - `35a6c86` (feat)

## Files Created/Modified

- `src/seer/config.py` - Typed schema, strict decoder, validation, JSON I/O, and digest helpers.
- `src/seer/cli.py` - Argument parser, typed invocation, overrides, and injectable dispatch.
- `tests/test_config.py` - Round-trip, normalization, digest, and strict failure tests.
- `tests/test_cli.py` - Command surface, override, fixture, and dispatch tests.
- `examples/synthetic.json` - Fully explicit Phase 1 synthetic configuration.
- `pyproject.toml` - Installed `seer` console script.

## Decisions Made

- Required dataset, seed, energy, and output identity groups explicitly while preserving existing model, optimizer, training, and evaluation field names.
- Kept the smoke handler late-bound for Plan 01-04 and made every unbound command fail explicitly.
- Applied CLI overrides to an immutable effective config using dataclass replacement.

## Deviations from Plan

None - plan scope and behavior were implemented as written.

## Issues Encountered

- The shared `.venv` could not be rebuilt offline because the newly resolved Transformers wheel was absent from the local uv cache. Focused tests were therefore run using the already cached pytest packages directly: **15 passed**. Ruff passed on every Plan 01-01 source and test file, manual CLI help/dispatch checks passed, and Python compilation passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plans 01-02 and 01-03 can consume the strict config and CLI invocation contracts.
- Plan 01-04 can register the synthetic smoke handler without changing parser behavior.

---
*Phase: 01-reproducible-experiment-runtime*
*Completed: 2026-07-16*
