---
phase: 02-multi-domain-evidence-data
plan: "02"
subsystem: evidence-data
tags: [datasets, adapters, provenance, offline, staging]
requires: [02-01, 01-reproducible-experiment-runtime]
provides:
  - Exact-revision GSM8K, ProofWriter, and bAbI adapter contracts
  - Explicit-consent atomic source staging with immutable hashes
  - Dataset source lock including full commit and runtime provenance
affects: [02-03, 02-04, phase-03, phase-04]
tech-stack:
  added: [datasets-3.6.0]
  patterns: [injected resolver-loader protocols, fail-closed source locks, atomic staging]
key-files:
  created: [src/seer/adapters.py, src/seer/preparation.py, tests/test_adapters.py, examples/evidence.json]
  modified: [src/seer/config.py, src/seer/cli.py, tests/test_cli.py, pyproject.toml, uv.lock]
key-decisions:
  - Existing synthetic v1 wire output omits empty real-data extensions for compatibility.
  - Source staging is immutable and never writes a COMPLETE marker.
  - Network-capable dataset imports and calls occur only beyond explicit operator consent.
patterns-established:
  - Tests inject source-shaped iterables and resolver facts without contacting the Hub.
  - Every staged normalized shard is verified against dataset-lock.json before loading.
requirements-completed: [DATA-02]
duration: 18 min
completed: 2026-07-16
---

# Phase 2 Plan 02: Pinned Dataset Adapters and Source Staging Summary

**SEER now converts pinned GSM8K, ProofWriter, and bAbI sources into canonical staged evidence
behind an explicit download-consent boundary, with full source and normalized-shard provenance.**

## Accomplishments

- Extended strict configuration with real dataset specifications and generation regimes while
  preserving the exact existing synthetic configuration wire representation.
- Added fixture-verified adapters with exact v1 prompts, stable source/group identities, strict
  gold validation, and prior-context-only bAbI question expansion.
- Added atomic, immutable source staging and a dataset lock containing full revisions, source file
  hashes, official split facts, schema, counts, license, fingerprint, library version, and output
  hashes.
- Registered `prepare-data --allow-download`; resolver access is impossible without the explicit
  flag, and no staging path writes final publication's `COMPLETE` marker.

## Task Commits

1. **Task 02-02-01: Add strict multi-dataset configuration and pinned row adapters**
   - `ce98390` (feat)
2. **Task 02-02-02: Build opt-in source staging and dataset preflight lock**
   - `a2430c0` (feat)

## Decisions Made

- ProofWriter sample caps are enforced while iterating, so staging stops consumption without
  materializing the remaining source corpus.
- Existing staging is verified but never overwritten; mutation, revision ambiguity, count/schema,
  split, license, or source substitution mismatches fail closed.
- The production Hugging Face backend imports `datasets` lazily after consent. Automated tests use
  injected resolvers/loaders exclusively.

## Deviations from Plan

- The manual one-row real-Hub preflight was not run because the plan marks it network-consented and
  the user did not authorize downloading dataset rows. All automated source fixtures remained
  offline as required.

## Verification

- Focused adapter/config/CLI suite: **26 passed**.
- Full offline suite: **130 passed** in 4.20 seconds.
- Ruff: **all checks passed**.
- Offline lock check: **84 packages resolved**, including locked `datasets==3.6.0`.

## User Setup Required

Real source preparation still requires an explicit invocation of `prepare-data --allow-download`.
Plan 02-03 owns partition/audit publication and the final `COMPLETE` marker.

## Next Plan Readiness

- Plan 02-03 can verify staged hashes, assign protected groups, audit leakage/corruption, and only
  then publish a completed prepared corpus.
- Plan 02-04 can rely on exact prompt payloads and source provenance without live Hub access.

## Self-Check: PASSED

---
*Phase: 02-multi-domain-evidence-data*
*Completed: 2026-07-16*
