---
phase: 02-multi-domain-evidence-data
plan: "01"
subsystem: evidence-data
tags: [schemas, canonical-json, normalization, gsm8k, proofwriter, babi]
requires: [01-reproducible-experiment-runtime]
provides:
  - Strict versioned scientific evidence records and canonical JSON/JSONL codecs
  - Stable example, generation, and failure identity helpers
  - Exact domain normalizers with explicit invalid and ambiguous outcomes
  - Traceable scoring from raw generation and source gold
affects: [02-02, 02-03, 02-04, phase-03, phase-04, phase-05]
tech-stack:
  added: []
  patterns: [frozen slots records, strict discriminated codec, exact rational normalization]
key-files:
  created: [src/seer/evidence.py, src/seer/normalization.py, tests/test_evidence.py, tests/test_normalization.py]
  modified: []
key-decisions:
  - Scientific evidence is split into four non-nullable record types with a record_type discriminator.
  - Rational answers use Fraction equality and canonical numerator/denominator text.
  - Invalid source gold yields null correctness; deterministic invalid predictions remain incorrect.
patterns-established:
  - Canonical compact sorted JSON is the sole hashing and wire representation.
  - Normalizers recognize only explicit source or FINAL markers and never fall back to arbitrary text.
requirements-completed: [DATA-01, DATA-03]
duration: 12 min
completed: 2026-07-16
---

# Phase 2 Plan 01: Evidence Records and Domain Normalization Summary

**SEER now has one strict scientific evidence boundary and deterministic, exact answer scoring for
GSM8K, ProofWriter, and bAbI without changing the existing tensor smoke interface.**

## Performance

- **Duration:** 12 min
- **Completed:** 2026-07-16
- **Tasks:** 2
- **Files created:** 4

## Accomplishments

- Added frozen, slotted `TaskExample`, `GenerationRecord`, `ScoredResult`, and `FailureRecord`
  schemas with strict schema-version, discriminator, enum, missing-field, unknown-field, and
  identifier validation.
- Added canonical JSON and JSONL codecs plus SHA-256 identities whose results are stable across
  mapping key order and sensitive to source, prompt, model, regime, seed, and decoding identity.
- Added NFKC-aware exact rational, categorical, and entity normalizers with structured failure
  outcomes for empty, missing, malformed, thinking-tag, and conflicting answers.
- Added traceable scoring that retains raw prediction, extracted candidate spans, canonical
  prediction and gold, correctness, and failure reason.

## Task Commits

1. **Task 02-01-01: Define strict canonical evidence records and identifiers**
   - `6b3adfd` (feat)
2. **Task 02-01-02: Implement explicit domain normalizers and scoring outcomes**
   - `2ec47b6` (feat)

## Files Created

- `src/seer/evidence.py` - Evidence schemas, strict codecs, and stable identity helpers.
- `src/seer/normalization.py` - Pure domain normalizers and score construction.
- `tests/test_evidence.py` - Wire, identity, purity, and smoke-interface contract tests.
- `tests/test_normalization.py` - Golden and adversarial normalization/scoring matrix.

## Decisions Made

- The wire format includes an explicit `record_type` discriminator while dataclass payloads retain
  only scientific fields; timestamps and machine paths remain outside these records.
- Multiple equal explicit answers collapse to one exact value; conflicting explicit answers are
  ambiguous, and malformed or unsupported forms are invalid.
- GSM8K decimals, fractions, commas, signed values, and percentages canonicalize through
  `fractions.Fraction`; expressions, tolerance, fuzzy matching, and last-number inference are not
  supported.

## Deviations from Plan

None.

## Issues Encountered

- The sandbox cannot write uv's default cache, so verification used
  `UV_CACHE_DIR=/tmp/seer-uv-cache` with `uv run --offline`; no data, weights, or dependencies were
  downloaded.
- Torch emitted environment-only NVML warnings during the full CPU suite; all tests passed.

## Verification

- Test-first RED for evidence records: collection failed because `seer.evidence` did not exist.
- Test-first RED for normalizers: collection failed because `seer.normalization` did not exist.
- `uv run --offline pytest tests/test_evidence.py tests/test_normalization.py -q` - **41 passed**.
- `uv run --offline pytest` - **119 passed** in 3.34 seconds.
- `uv run --offline ruff check .` - **all checks passed**.
- Existing `SyntheticStateTrackingDataset` tensor shape contract remains green and its source was
  not changed.

## User Setup Required

None. This plan is fully offline and does not prepare or download datasets.

## Next Plan Readiness

- Plan 02-02 can construct pinned source adapters directly into `TaskExample` records.
- Plans 02-03 and 02-04 can hash, partition, serialize, generate, and score using the established
  canonical contracts.

## Self-Check: PASSED

---
*Phase: 02-multi-domain-evidence-data*
*Completed: 2026-07-16*
