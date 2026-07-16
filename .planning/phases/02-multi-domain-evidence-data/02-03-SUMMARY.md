---
phase: 02-multi-domain-evidence-data
plan: "03"
subsystem: evidence-data
tags: [partitions, leakage, deduplication, corruptions, transactions]
requires: [02-01, 02-02, 01-reproducible-experiment-runtime]
provides:
  - Deterministic group-safe protected partition assignment
  - Fail-closed duplicate, conflict, and leakage audit
  - Prompt-only generation views with scoped gold access
  - Provenance-rich corruption records in a separate namespace
  - Transactional prepared-corpus publication with COMPLETE-last eligibility
affects: [02-04, phase-03, phase-04, phase-05]
tech-stack:
  added: []
  patterns: [integer hash partitions, capability-scoped labels, COMPLETE-last publication]
key-files:
  created: [src/seer/partitions.py, src/seer/corruptions.py, tests/test_partitions.py]
  modified: [src/seer/preparation.py, src/seer/cli.py, tests/test_adapters.py, tests/test_cli.py]
key-decisions:
  - Adapter partition placeholders are always replaced at the protected assignment boundary.
  - Exact content or group overlap across partitions aborts before publication.
  - COMPLETE is the only generation-eligibility marker; interrupted promotion remains ineligible.
requirements-completed: [DATA-04, DATA-06]
duration: 20 min
completed: 2026-07-16
---

# Phase 2 Plan 03: Protected Partitions and Publication Summary

**SEER now converts verified staged examples into deterministic protected partitions, audits and
quarantines source defects, separates constructed evidence, and publishes only hash-validated
corpora as generation eligible.**

## Accomplishments

- Implemented the exact first-64-bit SHA-256 partition protocol with official test preservation,
  GSM8K 70/15/15 development assignment, and group-safe ProofWriter/bAbI 50/50 validation splits.
- Added deterministic duplicate collapse, conflicting-gold quarantine, protected content/group
  overlap rejection, duplicate-source reporting, and cross-domain prompt overlap diagnostics.
- Added prompt/ID-only protected generation records and a separate gold-scoring capability.
- Added deterministic corruption fixtures with stable identities and full provenance, explicit-use
  capability, confirmatory-test prohibition, and natural-iterator rejection.
- Completed `prepare_data` publication of dataset lock, partition manifest, leakage audit,
  deduplicated example shards, quarantine, corruption shards, artifact manifest, and COMPLETE last.

## Task Commits

1. **Task 02-03-01: Implement group-safe partitioning and fail-closed leakage audit**
   - `b0f397f` (feat)
2. **Task 02-03-02: Separate corruption provenance from natural evidence**
   - `62908c4` (feat)
3. **Task 02-03-03: Publish the complete prepared corpus as one transaction**
   - `9937179` (feat)

## Decisions Made

- Canonical audit normalization is deliberately exact and conservative: NFKC, case, whitespace,
  line endings, and harmless terminal punctuation only.
- Same-partition duplicate winners use lexicographic source identity and preserve multiplicity in
  the leakage audit; every member of a conflicting-gold set is quarantined.
- Approximate similarity remains report-only and is not used to delete or reassign evidence.
- Partial publication bytes are harmless because consumers must require and validate COMPLETE.

## Deviations from Plan

- The real prepared corpus and manual audit inspection were not run because dataset-row downloads
  were not authorized. Production-shaped injected fixtures exercise the complete publication path
  offline.

## Verification

- `tests/test_partitions.py`: **6 passed** after corruption integration.
- Focused adapter/partition/CLI integration suite: **26 passed**.
- Full offline suite: **139 passed** in 3.63 seconds.
- Ruff: **all checks passed**.
- No model weights or dataset rows were downloaded.

## User Setup Required

Real corpus publication still requires explicit `prepare-data --allow-download` consent.

## Next Plan Readiness

- Plan 02-04 can consume only prompt-safe views, verify the leakage-audit hash, and open gold labels
  exclusively at its scoring boundary.
- Later sufficiency and evaluation code can reject corruption records mechanically.

## Self-Check: PASSED

---
*Phase: 02-multi-domain-evidence-data*
*Completed: 2026-07-16*
