---
phase: 01-reproducible-experiment-runtime
plan: "03"
subsystem: model-cache
tags: [qwen3, huggingface, offline, cache, reproducibility]
requires: [01-01]
provides:
  - Exact-revision local-only Qwen3 snapshot resolution
  - Essential-file and snapshot-commit verification
  - Typed snapshot metadata hashes for experiment manifests
  - Fail-closed local model loading with a Transformers version preflight
affects: [01-04]
tech-stack:
  added: []
  patterns: [injected resolver and loader, single-attempt offline resolution, metadata-only hashing]
key-files:
  created: [src/seer/cache.py, tests/test_cache.py]
  modified: [src/seer/model.py, tests/test_model.py, pyproject.toml, uv.lock]
key-decisions:
  - Snapshot identity is verified from the exact revision directory before model construction.
  - Small metadata and weight-index files are hashed, while large model weight bytes are not.
  - Online download remains an explicit operator action printed in diagnostics, never a fallback.
patterns-established:
  - Hugging Face boundaries accept injected resolver and loader callables for cache-independent tests.
  - Real model construction consumes a verified local path and always passes local_files_only=True.
requirements-completed: [EXP-06]
duration: 12 min
completed: 2026-07-16
---

# Phase 1 Plan 03: Offline Qwen3 Model Boundary Summary

**Pinned Qwen3 cache verification and local-only loading with actionable, fail-closed diagnostics**

## Performance

- **Duration:** 12 min
- **Completed:** 2026-07-16
- **Tasks:** 1
- **Files modified:** 6

## Accomplishments

- Added exact `Qwen/Qwen3-0.6B` snapshot resolution at commit
  `c1899de289a04d12100db370d81485cdf75e47ca`, with one local-only resolver call.
- Added typed resolved-snapshot provenance, essential config/tokenizer/weight checks, and SHA-256
  hashes for small metadata files suitable for later manifests.
- Changed `SeerPathAModel.from_pretrained` to validate Transformers >=4.51 and consume only a
  verified local snapshot while retaining direct fake-model construction.
- Raised the project and lock-file Transformers floor to >=4.51; the lock remains valid at its
  existing Transformers 4.68.4 resolution.

## Task Commits

1. **Task 01-03-01: Implement and test the pinned local model boundary**
   - `9af0bfd` (test)
   - `7032643` (feat)

## Files Created/Modified

- `src/seer/cache.py` - Local-only resolver, snapshot verifier, version preflight, hashes, and diagnostics.
- `src/seer/model.py` - Verified-snapshot loader boundary with injected test seams.
- `tests/test_cache.py` - Exact argument propagation and fail-closed cache/version/file tests.
- `tests/test_model.py` - Offline loader propagation and pre-construction rejection tests.
- `pyproject.toml` - Transformers >=4.51 dependency floor.
- `uv.lock` - Matching direct dependency constraint with Transformers 4.68.4 retained.

## Decisions Made

- Accepted either tokenizer JSON metadata file and either indexed or single-file safetensors/bin
  weights, matching supported Hugging Face snapshot layouts without weakening essential checks.
- Used an explicit `huggingface-cli download` command in errors as operator guidance only.
- Kept the real forward pass manual-only; automated tests use injected fakes and never load weights.

## Deviations from Plan

None - plan scope and behavior were implemented as written.

## Issues Encountered

- The sandbox could not write uv's default cache lock. Required commands were run with explicit
  offline mode and existing cached tools; no dependencies or weights were downloaded.

## Verification

- `uv run --offline pytest tests/test_model.py tests/test_cache.py -q` - **17 passed**.
- `uv lock --check` - **resolved 64 packages; lock is current**.
- `uv run --offline ruff check src/seer/cache.py src/seer/model.py tests/test_cache.py tests/test_model.py` - **passed**.
- Safe local-cache file preflight - **passed** for the exact approved snapshot and essentials.
- Real model forward pass - **not run**, per the plan's manual-only boundary.

## User Setup Required

None - no automatic download or external service configuration was introduced.

## Next Phase Readiness

- Plan 01-04 can record `ResolvedSnapshot.metadata_hashes` in runtime provenance and expose the
  cached-model preflight operationally.
- The synthetic smoke path remains wholly independent of Hugging Face model construction.

## Self-Check: PASSED

---
*Phase: 01-reproducible-experiment-runtime*
*Completed: 2026-07-16*
