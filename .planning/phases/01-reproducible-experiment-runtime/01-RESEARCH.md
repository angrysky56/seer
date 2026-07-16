# Phase 1 Research — Reproducible Experiment Runtime

## Scope

Phase 1 should turn the existing Path A library skeleton into an offline-capable,
resumable experiment runtime without implementing the real-domain adapters or the
full evidence pipeline assigned to later phases. It owns EXP-01 through EXP-06,
DOC-01, QUAL-01, QUAL-02, and QUAL-04.

The codebase already provides a good seam for this work: `src/seer/config.py`
contains nested typed dataclasses, `SeerPathAModel` accepts a fake base model for
tests, and the synthetic parity dataset plus `train_loop` already form a CPU-safe
vertical slice. The runtime should wrap these primitives rather than move domain,
signal, or statistical work forward from later phases.

## Existing-State Findings

- `ExperimentConfig` composes model, optimizer, training, and evaluation
  dataclasses, but there is no loader, validator, version field, canonical
  serialization, or output/runtime configuration.
- `SeerPathAModel.from_pretrained` calls `AutoModelForCausalLM.from_pretrained`
  with only a model name. It can access the network, does not pin a revision, and
  does not record the resolved snapshot.
- There is no console entry point or CLI module in `pyproject.toml`.
- `train_loop` returns in-memory step records and has no checkpoint or resume
  contract. It also relies on shuffled `DataLoader` state, so exact mid-epoch
  resumption cannot be inferred from a step count alone.
- Tests are already offline and use `tests/fakes.py`; this constraint should be
  preserved by dependency injection, not by mocking Hugging Face globally.
- `pyproject.toml` currently allows `transformers>=4.46`, below the Qwen3 floor
  fixed by the milestone (`>=4.51`).
- There are no experiment artifacts, schemas, hashes, atomic writers, run
  locks, completion markers, or operational docs.

## Recommended Architecture

### 1. Extend the typed configuration with an explicit schema boundary

Keep standard-library dataclasses as the source of truth to avoid introducing a
second model system. Add a top-level `schema_version` and dedicated runtime/output
configuration covering at least:

- command/stage selection and synthetic versus real backend;
- deterministic seed policy;
- output root, run name or optional explicit run ID;
- resume/replace policy;
- offline mode, model revision, cache location override, and local-files-only;
- checkpoint interval and artifact schema version.

Implement recursive `to_dict`/`from_dict` functions with strict validation:
reject unknown keys, missing required keys, invalid literals, malformed tuple/list
values, and unsupported schema versions with field-qualified errors. JSON should
be the required interchange format in Phase 1. YAML is not currently a project
dependency and is unnecessary for the acceptance criteria; it can be added later
if users need comments or anchors.

Canonical serialization should use UTF-8 JSON with sorted keys and stable compact
separators for hashing, while a separately indented form is written for humans.
Paths should serialize as strings and enums/literals as their wire values. Do not
serialize live device objects, model instances, or derived hardware facts into
the input config.

Recommended run identity is a content-derived `config_digest` plus a readable
slug, but timestamps must not participate in the digest. A repeated identical
config should resolve to the same logical run unless an explicit new run ID is
requested. Derived facts such as git revision and dependency versions belong in
the manifest rather than the run identity.

### 2. Use one thin CLI with stage subcommands

Add a package entry point such as `seer = "seer.cli:main"` and implement with
`argparse`; the standard library is sufficient and keeps the runtime lightweight.
Expose the milestone command surface now even if later-phase commands initially
fail with a clear “not implemented in this phase” diagnostic:

- `prepare-data`
- `cache-outputs`
- `train`
- `evaluate`
- `run-matrix`
- `build-report`
- `smoke`

All commands should accept a config path and narrowly scoped runtime overrides
(`--output-root`, `--resume`, `--replace`, `--offline`). Avoid arbitrary dotted
config overrides in Phase 1 because they make the exact effective configuration
harder to audit. Always persist the fully resolved effective config before work.

The `smoke` command should be the Phase 1 vertical slice: create the fake model,
synthetic dataset, optimizer, a very short CPU training run, and production-shaped
result records. It must never instantiate an AutoTokenizer/AutoModel or dataset
loader. Real-model commands should cross an explicit loader boundary.

### 3. Make cached Qwen3 resolution explicit and fail closed

Represent the fixed model as:

- repository ID: `Qwen/Qwen3-0.6B`;
- revision: `c1899de289a04d12100db370d81485cdf75e47ca`;
- offline/local-only: true for the primary local experiment.

Use Hugging Face's cache resolver (or `snapshot_download` in local-only mode) to
resolve and verify the exact snapshot before model construction, then pass the
resolved local snapshot directory to `AutoModelForCausalLM.from_pretrained` with
`local_files_only=True`. Also set `revision` when resolving by repository ID.
This separates cache verification from model loading and makes the resolved path
and commit hash available to the manifest.

Cache verification should check that the snapshot resolves to the requested
commit and that essential configuration/tokenizer/weight files are present. It
should not hash all 0.6B model bytes on every start; record the pinned commit and
hash the small metadata files, with an optional expensive integrity mode if later
needed. Catch cache-miss/offline exceptions and report the exact repository,
revision, cache root searched, and an explicit opt-in download command. Never
retry without `local_files_only`, and never silently change revisions.

Update the dependency constraint and lock file to `transformers>=4.51`. A runtime
preflight should inspect `transformers.__version__` and produce an actionable
error before loading when it is too old. Unit tests should inject a resolver and
loader fake to assert that offline flags and the exact revision are propagated;
the cached real model is suitable for an explicit local smoke command, not CI.

### 4. Treat a run directory as a small transactional state machine

Recommended layout:

```text
runs/<slug>-<config_digest>/
  config.json
  manifest.json
  state.json
  checkpoints/
  artifacts/
  logs/
  COMPLETE
```

Use explicit states such as `created`, `running`, `interrupted`, `failed`, and
`complete`, plus per-stage status. `COMPLETE` is the final commit marker and must
be written only after all required artifacts and hashes validate. A completed
directory is immutable by default:

- same config + no flags: report already complete and exit successfully;
- `--resume`: resume only incomplete compatible state;
- incompatible config or schema: fail with a diff/diagnostic;
- `--replace`: require explicit intent, preserve or rename the old run rather
  than mutating it in place where practical, then create a fresh transaction.

Prevent concurrent writers with an atomic lock-file creation (`O_CREAT|O_EXCL`)
that records PID, hostname, and start time. A stale lock must not be removed
automatically merely because its PID is absent on another host; provide a
deliberate recovery option and record it.

Every JSON/checkpoint write should go to a temporary file in the destination
directory, flush and `fsync`, then `os.replace` to the final name. Directory
`fsync` is desirable on POSIX after replacement. Artifact records should include
relative path, byte size, media/schema type, and SHA-256. Compute hashes from
final bytes and build the final manifest only after validation. Exclude mutable
state, locks, temporary files, and the manifest's own hash from the manifest
artifact list to avoid self-reference.

### 5. Define deterministic resumption at stage boundaries first

For Phase 1, guarantee exact deterministic replay and resumption at atomic stage
or checkpoint boundaries, not at arbitrary Python instruction boundaries. Each
checkpoint should save:

- model and optimizer state;
- completed global step and next batch/epoch position;
- Python, NumPy, Torch CPU, and CUDA RNG states when applicable;
- sampler/generator state or a deterministic epoch permutation;
- effective config digest and checkpoint schema version.

Seed Python, NumPy, and Torch centrally. Construct the DataLoader with an explicit
`torch.Generator`, avoid worker processes in the CPU smoke test, and make example
order a pure function of seed/epoch or persist the sampler state. The current
`train_loop` will need a resumable runner seam or callbacks; do not infer exact
resume from `step` while allowing DataLoader shuffle to restart.

The smoke acceptance test should compare canonical result/artifact payload hashes
between a clean run and an interrupted-then-resumed run. Volatile manifest fields
(wall-clock timestamps, PID, hardware timing) should be excluded from that
scientific-equivalence comparison. “Identical validated records” should mean the
same canonical scientific records and artifact schema, not byte-identical
operational metadata.

### 6. Separate provenance from scientific artifacts

The initial manifest schema should include:

- schema version, logical run ID, status, command, effective config and digest;
- git HEAD and dirty flag (plus a diff hash when dirty, without embedding the
  potentially sensitive diff);
- Python/platform, dependency versions, hardware/device information;
- requested model ID/revision and resolved snapshot;
- dataset identifiers/revisions (empty or synthetic descriptor in Phase 1);
- all declared seeds and stage timestamps;
- artifact inventory containing SHA-256 and size;
- parent/resume checkpoint and replacement/recovery events.

Keep generated results in versioned JSON/JSONL records under `artifacts/` and use
the same schema in synthetic and real runs. Phase 1 can define a minimal envelope
(`schema_version`, run/stage IDs, status, records, diagnostics) that later phases
extend without changing the runtime contract.

### 7. Documentation and dependency work

Add an operations document (or a focused README section) with:

- `uv sync`, offline test/lint commands, and the Transformers Qwen3 floor;
- explicit opt-in model download and verification commands;
- synthetic CPU smoke invocation and expected artifact tree;
- real cached-model preflight that does not download;
- resume, completed-run no-op, replacement, stale-lock recovery, and artifact
  verification examples;
- interpretation of config digest, manifest hashes, status, and `COMPLETE`;
- warning that unit tests never access external models or datasets.

The lock file must be refreshed by `uv` after the constraint change. Avoid adding
large runtime libraries solely for CLI/config concerns.

## What Not to Build in Phase 1

- Do not download or adapt GSM8K, ProofWriter, or bAbI (Phase 2).
- Do not implement new energy variants, baselines, calibration, bootstrap
  statistics, or gate decisions (Phases 3–6).
- Do not make CI depend on the local Hugging Face cache or GPU.
- Do not claim cross-machine bitwise determinism; record hardware/software and
  test deterministic CPU equivalence in the supported environment.
- Do not use timestamps as identity or include them in scientific artifact
  equality checks.
- Do not overwrite completed evidence merely because a path exists.

## Likely Implementation Slices

1. **Config and CLI contract:** versioned strict JSON serialization, runtime
   settings, CLI entry point, config-focused tests.
2. **Transactional run store:** canonical run identity, atomic JSON/artifact
   writes, locks, state transitions, hashing, manifest provenance, no-overwrite
   and resume compatibility tests.
3. **Offline model boundary:** Qwen3 dependency floor/lock, exact cached-snapshot
   resolver, local-only loader propagation, actionable cache diagnostics, fake
   resolver tests.
4. **Synthetic vertical slice and docs:** deterministic CPU smoke pipeline,
   interruption/resumption equivalence, production-shaped artifacts, offline
   full suite and operational documentation.

This ordering lets each slice end with a user-visible/testable capability and
keeps the offline loader independent of the synthetic CI path.

## Validation Architecture

### Test layers

**Unit tests (fast, always offline)**

- strict config round-trip, canonical digest stability, unknown/missing field and
  schema-version failures;
- CLI parsing and actionable exit codes/messages;
- atomic-write behavior under an injected failure before replacement;
- artifact SHA-256/size verification and tamper detection;
- run-state transition rules, lock contention, completed-run immutability,
  explicit replacement, incompatible-resume rejection;
- model resolver receives exact Qwen repository, commit, cache root, and
  `local_files_only=True`; cache miss never invokes a network fallback;
- RNG capture/restore and deterministic sampler continuation.

Use `tmp_path`, monkeypatch/injected clocks, fake provenance collectors, and fake
model/cache resolvers. No test should assume the developer's home-cache layout.

**Integration tests (CPU, synthetic only)**

- invoke the CLI as a subprocess or `main(argv)` with a tiny JSON config;
- verify the expected production-shaped config, state, result, manifest,
  artifact hashes, and `COMPLETE` marker;
- rerun the completed command and verify no files are overwritten;
- inject interruption after a checkpoint, resume, and compare canonical
  scientific records/hashes against an uninterrupted reference run;
- corrupt/delete an artifact and verify validation fails closed;
- block socket/network calls during the suite to prove the smoke path is offline.

**Explicit local acceptance checks (not CI)**

- verify the exact cached Qwen3 snapshot and load config/tokenizer/model with
  local-only flags;
- confirm a deliberately missing snapshot produces the documented command and
  performs no network request;
- optionally run a very small cached-model forward pass on available hardware.

### Requirement-to-evidence matrix

| Requirement | Primary verification evidence |
|---|---|
| EXP-01 | strict serialization/validation unit tests and checked-in example config |
| EXP-02 | CLI help/dispatch tests plus synthetic command integration test |
| EXP-03 | manifest schema, provenance fields, hash verification and tamper test |
| EXP-04 | no-overwrite, compatibility, interruption/resume equivalence tests |
| EXP-05 | network-blocked full pytest suite using only fakes/synthetic data |
| EXP-06 | dependency/lock assertion, exact-revision resolver tests, local acceptance preflight |
| DOC-01 | documented commands exercised by CLI tests and acceptance checklist |
| QUAL-01 | `uv run pytest` with network disabled or guarded |
| QUAL-02 | `uv run ruff check .` |
| QUAL-04 | CPU CLI E2E creates validated production-shaped artifacts |

### Nyquist cadence

Run focused tests after each slice, then `uv run pytest` and
`uv run ruff check .` at the phase gate. The final gate should additionally run
the smoke command twice plus one interrupted/resumed scenario in separate
temporary run roots and validate artifacts using the runtime's own verifier.
Do not use the cached real checkpoint as evidence for QUAL-04.

### Fail-closed acceptance conditions

- No `COMPLETE` marker exists unless all declared artifacts validate.
- Any config-digest, checkpoint-schema, artifact-hash, or requested-snapshot
  mismatch prevents resume/promotion.
- A cache miss in offline mode exits before model construction and exposes no
  automatic online retry.
- A completed run remains byte-for-byte unchanged on an ordinary rerun.
- The test suite passes when network access is unavailable.

## Workflow Tooling Defect (Planning Context)

The installed GSD helper at
`/home/ty/.claude/gsd-core/bin/gsd-tools.cjs` currently fails before command
execution. Its dependency `bin/lib/runtime-artifact-conversion.cjs` requires
`../../../package.json`, but that file is absent, producing
`Error: Cannot find module '../../../package.json'`.

This affects the orchestration workflow's helper commands and should be recorded
by the planner as tooling context or handled manually. It is not evidence of a
SEER runtime defect and is not automatically in Phase 1 implementation scope.

## Key Recommendations

- Preserve dataclasses and add strict, versioned JSON at the runtime boundary.
- Make run identity content-derived and completed runs immutable by default.
- Build all writes and state transitions transactionally, with hashes and an
  unambiguous final `COMPLETE` marker.
- Guarantee deterministic checkpoint-boundary resume by persisting RNG and data
  ordering state.
- Resolve the exact Qwen3 commit locally first, then load from its resolved path;
  never permit an offline fallback to the network.
- Use the fake model and synthetic parity dataset for the production-shaped CPU
  E2E; keep real cached-model checks explicit and outside CI.
- Raise and lock Transformers to at least 4.51, then document preflight,
  download, smoke, resume, replacement, and artifact verification operations.

## RESEARCH COMPLETE
