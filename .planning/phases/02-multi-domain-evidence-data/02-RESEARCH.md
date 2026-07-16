# Phase 2 Research — Multi-Domain Evidence Data

## Scope

Phase 2 should turn three public reasoning datasets into immutable, normalized
example and generation corpora. It owns DATA-01 through DATA-07. It should not
fit a signal, implement baselines, calculate AUROC, or inspect confirmatory labels
for model selection; those belong to later phases.

The user delegated prompt, revision, normalization, and partition choices to this
research. The recommendations below therefore form the default protocol unless a
preparation-time compatibility check fails before confirmatory generations exist.

## Existing-State Findings

- `src/seer/data.py` currently models dense synthetic state-tracking tensors. It
  is a training interface, not an evidence-record interface. Preserve it for the
  smoke path and add a separate real-task record layer rather than forcing text
  examples into tensor fields.
- `DatasetConfig` has one dataset name/revision and split strings. Phase 2 needs a
  collection of pinned dataset specifications plus generation regimes. This is a
  configuration schema change and must remain strict and versioned.
- `seer prepare-data` and `seer cache-outputs` are declared but deliberately
  unimplemented. They are the natural explicit network and model-generation
  boundaries.
- Phase 1 already provides atomic run storage, manifest dataset provenance,
  immutable completion, hashing, offline Qwen resolution, and JSON/JSONL result
  envelopes. Reuse those contracts.
- The dependency set does not contain Hugging Face `datasets`. Add it as a runtime
  dependency and lock the exact resolved version. Tests must inject in-memory rows
  and never invoke Hub loading.
- The local target is one RTX 3060 (12 GB) and 64 GB RAM. Qwen3-0.6B generation is
  feasible one model at a time, but a full ProofWriter materialization is much
  larger than needed. Adapters should load only declared splits/columns, stream or
  shard preparation, and stop once predeclared sample caps are reached.
- The GSD helper's package metadata currently has a `package.json` defect observed
  by the orchestrator. This is workflow context only and must not influence the
  SEER implementation or evidence protocol.

## Primary Sources and Stable Facts

- Hugging Face Datasets supports a `revision` argument for commit/tag/branch
  selection, and assigns fingerprints to cached Arrow data. Record both the Hub
  commit and the resulting dataset fingerprint; a fingerprint alone is not a
  durable source revision. See the official loading and cache documentation:
  https://huggingface.co/docs/datasets/loading and
  https://huggingface.co/docs/datasets/en/about_cache
- `openai/gsm8k` has `main` and `socratic` configurations, 7,473 training and
  1,319 test examples (the current viewer/card sometimes labels the second split
  “validation”; the adapter must inspect and lock the actual split names), fields
  `question` and `answer`, and an MIT license:
  https://huggingface.co/datasets/openai/gsm8k
- `tasksource/proofwriter` exposes `train`, `validation`, and `test`, with fields
  including `theory`, `question`, `answer`, and `config`. Its card is sparse and
  does not declare a license. Treat the license as `unknown/not declared`; do not
  redistribute its rows. The Hub repository currently reports 585,552 train,
  85,468 validation, and 174,476 test rows:
  https://huggingface.co/datasets/tasksource/proofwriter
- `facebook/babi_qa` provides task-specific configurations, including
  `en-valid-10k-qa1` through `qa20`, with official train/validation/test splits.
  Its structured `story` contains context and question lines, supporting IDs, and
  answers. The dataset card declares CC BY 3.0:
  https://huggingface.co/datasets/facebook/babi_qa/blob/main/README.md
- The Qwen3 model card says `enable_thinking=False` strictly disables thinking,
  while thinking mode emits a `<think>...</think>` block. It recommends sampling
  `(temperature=0.6, top_p=0.95, top_k=20)` for thinking and
  `(temperature=0.7, top_p=0.8, top_k=20)` for non-thinking; it also recommends a
  standardized boxed final answer for math benchmarks:
  https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/README.md

## Recommended Dataset Lock

Use a checked-in logical lock specification and produce a preparation-time
`dataset-lock.json` containing the full resolved 40-character commit, selected
files and SHA-256 hashes, builder/config, official split names and counts,
`datasets` fingerprint, features, license declaration, and normalized-output
hashes. Never accept `main` as the evidence identity.

| Domain | Repository/config | Requested revision | Official source use |
|---|---|---|---|
| GSM8K | `openai/gsm8k`, `main` | `740312a` | Official train is development material; official test is confirmatory |
| ProofWriter | `tasksource/proofwriter`, `default` | `761ca6eedf37f1c27a4eeb88cd5107ada469a4ec` | Train/validation are development; test is confirmatory |
| bAbI | `facebook/babi_qa`, `en-valid-10k-qa1`, `qa2`, `qa3` | `ab3777b46c6c0d9a4513cd3b82ea6562293837a8` | Official train/validation are development; official test is confirmatory |

Rationale and safeguards:

- GSM8K's latest visible commit is abbreviated on the Hub. Resolve `740312a` to
  its full commit during explicit preparation and reject an ambiguous or changed
  resolution. If the abbreviated revision cannot resolve, stop and require the
  operator to update the checked-in lock before any generations; never fall back
  to `main`.
- The ProofWriter pin corresponds to its published parquet update. Because the
  card lacks a license and detailed provenance, cache only what is required and
  publish metadata/results, not raw rows.
- The bAbI pin is a known full repository commit with dataset builder version
  1.2.0 and a recorded upstream archive checksum. Newer Datasets versions may no
  longer execute legacy dataset scripts. Preparation must first try safe Hub
  parquet resolution for the exact logical dataset; if exact-revision parquet is
  unavailable, use a separately pinned compatible Datasets loader only in the
  opt-in preparation environment. Record the builder code hash, archive SHA-256
  (`0364ebde659f14d11bc21744516c5ec49d3d06cb692733f66680771244998898`),
  and library version. Do not silently substitute a different commit or source.
- Start bAbI with qa1/qa2/qa3 because they isolate one-, two-, and three-supporting
  fact entity-location tracking. Preserve `task_no` in every record. If natural
  class sufficiency fails, report underpowered. Do not expand to easier/different
  bAbI tasks after viewing signal results. A future protocol amendment may add a
  separately labeled exploratory corpus, never repair the primary corpus in place.

`prepare-data` is the only network-capable data command. Require an explicit
`--allow-download` (or equivalent configuration false by default), resolve and
verify revisions, then write immutable normalized shards. All later commands load
those shards with offline mode and verify their hashes. Document `hf download
<repo> --repo-type dataset --revision <pin>` as an optional manual prefetch, but
the application must still verify the result.

Recommended dependency policy: add `datasets>=3.2,<4` initially because the bAbI
pin uses a legacy builder, then let `uv.lock` pin the concrete version. Before
implementation, run a tiny explicit-download preflight for one row from every
config. If bAbI cannot load safely, prefer exact-revision parquet materialization
over widening the dependency range or enabling arbitrary remote code.

## Canonical Evidence Schemas

Use frozen dataclasses plus strict JSON codecs, `schema_version=1`, UTC-free
scientific payloads, and enums/Literals. Operational timestamps remain in the run
manifest. Define four related records rather than one nullable catch-all.

### `TaskExample`

- `example_id`: SHA-256 of canonical source identity, not row order alone;
- `domain`: `gsm8k | proofwriter | babi`;
- `dataset_id`, `dataset_revision`, `dataset_config`, `source_split`;
- `source_row_id` and `group_id` (story/theory/problem group);
- `partition`: `signal_train | model_selection | calibration | confirmatory_test`;
- `prompt_template_id` and `prompt_payload` (structured fields used to render);
- `prompt_text` (exact rendered user text before chat templating);
- `gold_raw`, `gold_normalized`, `answer_type`;
- `adapter_metadata` (ProofWriter depth/config; bAbI task/supporting IDs; GSM8K
  source answer extraction diagnostics);
- `content_fingerprint` and `group_fingerprint`;
- `license_id` and `corruption=None` for primary examples.

### `GenerationRecord`

- `generation_id`, `example_id`, model repository/revision, tokenizer revision;
- `chat_template_hash`, `prompt_token_ids_hash`, `prompt_token_count`;
- `regime`: `non_thinking | thinking`, and explicit `thinking_enabled`;
- complete generation parameters: sampling flag, seed, temperature, top-p,
  top-k, min-p if supported, repetition/presence penalty, max-new-tokens,
  stopping token IDs, padding/eos IDs, dtype and device;
- `raw_generation`, `raw_generated_token_ids` or their immutable artifact
  reference/hash, `generated_token_count`, finish reason, and truncation flag;
- separated `thinking_text` and `answer_text` for thinking mode, while retaining
  the unmodified raw string;
- stage/model errors represented by a linked failure record, never omitted rows.

### `ScoredResult`

- `generation_id`, `prediction_raw`, `prediction_normalized`, `gold_normalized`;
- `is_correct: bool | null`, `score_status: scored | invalid | ambiguous`;
- `normalizer_id/version`, extracted candidate spans, and `failure_reason`;
- no energy, baseline, calibration, or evaluation fields yet.

### `FailureRecord`

- `record_id`, `stage`, `code`, `message`, `retryable`, linked example/generation;
- structured context without stack traces or machine paths in scientific JSON;
- codes: `dataset_resolution_failed`, `schema_mismatch`, `source_gold_invalid`,
  `prompt_over_budget`, `generation_error`, `empty_generation`,
  `missing_final_answer`, `multiple_conflicting_answers`, `invalid_answer_type`,
  `thinking_parse_error`, `token_budget_exhausted`, and `scoring_error`.

Invalid/ambiguous generations remain in the corpus and count as naturally
incorrect for accuracy only when a deterministic scoring rule establishes that
they do not match gold. `is_correct=null` is reserved for invalid source gold or
an unevaluable infrastructure failure; those rows are excluded with explicit
counts, not converted to wrong answers.

## Prompt Protocol

Render one system message and one user message through the exact cached tokenizer
chat template with `add_generation_prompt=True`. Record the tokenizer/chat-template
hash and `enable_thinking` flag. Never hand-build Qwen control tokens.

System message for all domains:

> You solve the task using only the information in the user message. Follow the
> requested final-answer format exactly. Do not add facts.

Domain user templates, version `v1`:

- **GSM8K:** present the question verbatim, then: “Reason step by step. End with
  exactly `FINAL: \\boxed{answer}` where `answer` is only the final number.”
- **ProofWriter:** present `Facts and rules:\n{theory}\n\nClaim:\n{question}`, then:
  “Using open-world reasoning, answer whether the claim is entailed, contradicted,
  or unknown. End with exactly `FINAL: true`, `FINAL: false`, or
  `FINAL: unknown`.” The adapter must verify the source label mapping from golden
  rows rather than assume casing or vocabulary.
- **bAbI:** render the context lines preceding one question, then the question.
  End with: “Answer from the story. End with exactly `FINAL: answer` and no other
  text after it.” Each question in a story becomes an example whose context is
  only the prior lines, never future questions/answers.

Primary non-thinking generation is deterministic greedy decoding
(`do_sample=False`, one return sequence) with `enable_thinking=False`. This gives
one stable natural prediction per example and avoids turning generation noise
into the primary class label. Use domain caps of 256 new tokens for GSM8K, 96 for
ProofWriter, and 48 for bAbI. Reject prompts exceeding model context minus their
cap before generation.

Secondary thinking generation is a separately labeled, predeclared subset and
never pooled with primary data: `enable_thinking=True`, `do_sample=True`,
temperature 0.6, top-p 0.95, top-k 20, min-p 0 where supported, one return
sequence, recorded per-example seed, and `max_new_tokens=1024`. This follows the
model card's sampling recommendation while bounding local cost. Token-budget
exhaustion is a first-class failure and cannot be repaired by an unrecorded retry.
The secondary subset should be selected by example hash before any prediction is
seen (recommend 256 examples per domain, subject to split size).

Qwen recommends sampling parameters for non-thinking mode, but deterministic
greedy decoding is preferable for the primary measurement because this phase is
constructing a stable correctness corpus, not optimizing benchmark accuracy.
Later self-consistency and disagreement baselines own sampled generations.

## Domain Normalization and Correctness

All normalizers must be pure, versioned functions. Normalize Unicode with NFKC,
normalize line endings, and parse only a final-answer marker (plus tightly scoped
legacy fallbacks in golden tests). Never use Python `eval`.

### GSM8K numeric normalizer

1. Extract `FINAL: \\boxed{...}`; accept an unboxed `FINAL:` only as a recorded
   format deviation. For gold, extract the value after the dataset's final `####`.
2. Strip currency symbols, commas used as thousands separators, surrounding
   whitespace, and a terminal period. Reject embedded units unless the remaining
   value is unambiguous.
3. Parse signed integers, finite decimals, simple fractions, and percentages into
   an exact `fractions.Fraction` canonical `numerator/denominator` form. Reject
   NaN, infinity, expressions, and division by zero.
4. If multiple final markers yield unequal values, return
   `multiple_conflicting_answers`; do not choose the last arbitrary number.
5. Correctness is exact rational equality. Do not introduce numeric tolerances for
   a dataset whose gold answers are exact.

### ProofWriter categorical normalizer

- Case-fold and strip surrounding whitespace/punctuation from the final token.
- Map only predeclared synonyms: `true/entailed/yes`,
  `false/contradicted/no`, `unknown/both unknown/not provable` after verifying
  the dataset's source labels. Canonical values are `true`, `false`, `unknown`.
- Contradiction is not the same as absence of proof. Explanatory text containing
  more than one category without one unique `FINAL:` is ambiguous.

### bAbI entity normalizer

- Extract `FINAL:`, apply NFKC, Unicode case-fold, trim whitespace and surrounding
  sentence punctuation, and collapse internal whitespace.
- For qa1–qa3, require one non-empty entity/location phrase and compare exactly.
  Do not apply stemming, fuzzy matching, or synonym tables.
- Keep a generic extension point for later list/count/yes-no tasks, but do not
  implement or silently use it in the primary qa1–qa3 protocol.

Golden fixtures must include valid formatted answers, source-gold forms, negative
numbers, commas, decimals/fractions, duplicate equal markers, conflicting markers,
empty output, truncated boxes, thinking tags with/without final text, category
synonyms, punctuation/casing, and adversarial explanatory text.

## Protected Partitions

Preserve official boundaries first. `confirmatory_test` is always the official
test split and its labels are not exposed through development iterators or summary
APIs. Never repartition official test to repair class balance.

- **GSM8K:** deterministically group-hash official train into 70% signal train,
  15% model selection, 15% calibration. Official test is confirmatory.
- **ProofWriter:** official train is signal train; split official validation by
  group hash 50/50 into model selection and calibration; official test is
  confirmatory. Group by normalized theory (not individual question) so questions
  over one theory cannot cross partitions.
- **bAbI:** official train is signal train; split official validation 50/50 into
  model selection and calibration; official test is confirmatory. Group by the
  entire source story/config, not individual question.

Use `sha256("seer-partition-v1\0" + dataset_revision + "\0" + domain + "\0" +
group_fingerprint)`, interpret the first 64 bits as an unsigned integer, and apply
fixed integer thresholds. Do not use Python's randomized `hash`, row order,
`train_test_split`, or a seed-dependent shuffle. Write a partition manifest with
algorithm ID, thresholds, counts, group hashes, and artifact hashes.

The same protected partitions exist for every generation regime. Regime or seed
must never cause an example to move. For the primary transfer direction, only
bAbI `signal_train/model_selection/calibration` may fit/select/calibrate the
cross-domain signal. ProofWriter and GSM8K development partitions may support
adapter tests and explicitly labeled exploratory/domain-matched work, but later
code must require a protocol capability before labels are opened. Confirmatory
iterators should return prompts/IDs without gold until the scoring stage.

## Duplicate and Leakage Audit

Create two canonical hashes:

- `content_fingerprint`: domain + canonical task inputs + canonical gold;
- `group_fingerprint`: all examples sharing a problem/theory/story context.

Canonicalization for audit may normalize NFKC, case, whitespace, line endings,
and harmless terminal punctuation, but must not perform semantic paraphrasing.
Audit exact duplicate content, input-only duplicates with conflicting gold, group
overlap, duplicate source IDs, and cross-domain exact prompt overlap.

Policy:

- any content or group crossing protected partitions is a hard preparation
  failure;
- duplicates within one partition are reported and deterministically collapsed
  to the lexicographically smallest source identity, with multiplicity recorded;
- input-identical examples with conflicting gold are quarantined as source errors;
- approximate similarity is a diagnostic only (for example token 5-gram MinHash)
  and must not silently delete evidence;
- audit before generation and include the audit hash in every generation run.

## Natural Errors and Class Sufficiency

Generate and score every selected primary confirmatory example once before signal
fitting. Report per `(domain, regime, partition)` counts for correct, incorrect,
ambiguous, invalid-source, generation failure, and truncation. A valid formatted
or unformatted but deterministically wrong answer is a natural incorrect example.
Constructed corruptions are never used to satisfy natural-error counts.

The confirmatory AUROC eligibility rule is mechanical: at least 100 naturally
correct and 100 naturally incorrect scored generations in each domain/regime.
Otherwise emit `underpowered` with observed counts and recommended additional
evidence, and prohibit later evaluators from producing a conclusive AUROC. Do not
resample examples, generate repeatedly until an error appears, merge thinking and
non-thinking, or expand dataset/task scope after observing the balance.

Because bAbI qa1–qa3 may be too easy and GSM8K may be mostly wrong for a 0.6B
model, class imbalance is an expected scientific result. Phase 2 succeeds by
detecting it honestly. If a domain is underpowered, any amended exploratory
protocol must receive a new protocol/version ID and cannot replace the frozen
primary corpus.

## Corruption Provenance

Define a `CorruptionRecord` linked to an unchanged base example:

- `corruption_id`, `base_example_id`, strategy, strategy version, seed;
- structured parameters and before/after content hashes;
- intended label/use (`signal_training` or `ablation`, never primary test);
- generator code revision and validation status.

Store corruptions in separate artifacts/iterators and require an explicit
`include_corruptions` capability. Primary natural-result builders must reject any
record whose corruption field is non-null. Phase 2 only establishes schemas and
one or two deterministic fixture strategies sufficient to prove separation; the
semantically hard negative implementation belongs to Phase 3.

## Record Generation and Local Operation

Recommended preparation pipeline:

1. Resolve and download each explicit dataset revision with operator consent.
2. Validate license metadata, features, split names/counts, and source gold.
3. Stream source rows through adapters into canonical example shards; never load
   all ProofWriter proof text into memory when only theory/question/answer and
   small metadata are required.
4. Deduplicate, partition, audit, hash, and atomically finalize data artifacts.
5. In a separate offline `cache-outputs` run, verify the data lock and Qwen lock,
   render prompts, generate in bounded batches, normalize/score, and checkpoint at
   shard boundaries. Batch size 1–4 is appropriate for the RTX 3060; dynamically
   batch only equal-ish prompt lengths and record the batch policy.
6. Finalize JSONL shards plus an index/sufficiency report. Resume may only append
   missing generation IDs and must reject changed tokenizer, prompt, parameters,
   model, or data-audit hashes.

Keep raw text in local immutable artifacts because DATA-03 requires it. Reports
should refer to hashes and aggregate counts rather than redistribute dataset rows.
Generated token count is `len(output_ids) - input_length`, not a tokenizer count
of decoded text. Finish reason should distinguish EOS, length, error, and manual
stop. Store sampling seed even for greedy decoding (marked non-operative).

## Likely Implementation Slices

1. **Schema and normalization core:** strict records/codecs, failure taxonomy,
   three pure normalizers, fixtures, and production-shaped synthetic compatibility.
2. **Pinned adapters and preparation:** dataset lock/resolver boundary, three row
   adapters, explicit download/offline policy, source validation and manifests.
3. **Partitions and leakage:** group-aware deterministic assignment, duplicate
   audit, corruption separation, protected-access API, and fail-closed reports.
4. **Qwen generation corpus:** prompt registry, regime/token contracts, resumable
   local generation, scoring, class-sufficiency report, and operations docs.

This ordering gives the generation plan stable records and audited immutable
inputs. The dataset adapters and real-model runner should remain independently
testable through injected rows/tokenizer/model fakes.

## What Not to Build in Phase 2

- No hidden-state capture, concept projection, energy head, disagreement, geometry,
  learned probe, token-probability baseline, or calibration.
- No AUROC/ECE/bootstrap computation and no Gate A/B decision.
- No signal fitting on ProofWriter/GSM8K labels in the primary transfer protocol.
- No automatic network access, model/data downloads from tests, or redistribution
  of raw ProofWriter rows.
- No adaptive prompt tuning after confirmatory outcomes and no silent increase of
  token budgets or repeated generation to manufacture class balance.

## Validation Architecture

### Test layers

**Unit tests (fast and always offline)**

- strict round-trip and unknown/missing/version failures for all four schemas;
- stable example/generation IDs and canonical JSON across process restarts;
- GSM8K rational, ProofWriter categorical, and bAbI entity golden fixtures,
  including malformed, conflicting, ambiguous, and truncated outputs;
- prompt snapshots and chat-template argument propagation with a fake tokenizer;
- deterministic group partition thresholds independent of row order;
- exact duplicate, conflicting-gold, and protected-group leakage detection;
- corruption records rejected from natural-result iterators;
- generated-token count, regime metadata, budget exhaustion, and failure codes;
- class sufficiency at 99/100 boundaries and exclusion of null/unscored rows.

**Integration tests (offline fixtures/fakes)**

- each adapter transforms small committed source-shaped rows into canonical
  examples while preserving official split/config/source identity;
- shuffled input order yields byte-equivalent partition/audit artifacts;
- fake Qwen tokenizer/model runs non-thinking and thinking paths, resumes at a
  shard boundary without duplicate records, and emits production schemas;
- a synthetic mixed corpus produces natural/corruption-separated JSONL and a
  correct sufficiency report;
- complete artifact mutation or data-lock mismatch fails before generation.

**Manual/explicit-network checks**

- with `--allow-download`, resolve each requested revision to a full commit,
  inspect features/splits/counts/license, and record source file hashes/fingerprint;
- verify one row from every bAbI config and a small ProofWriter/GSM8K sample before
  the full preparation; ensure no arbitrary remote code or unexpected host is used;
- offline after preparation, load the exact cached Qwen snapshot and generate one
  golden example per domain/regime on the RTX 3060; inspect prompt formatting,
  token counts, thinking separation, and peak VRAM;
- run the full primary corpus only after the lock/audit artifact is reviewed.

### Requirement-to-evidence map

| Requirement | Automated evidence | Manual/real evidence |
|---|---|---|
| DATA-01 | strict schema/codec/ID tests; synthetic compatibility | inspect JSONL/index schema |
| DATA-02 | source-shaped adapter fixtures and split-preservation tests | exact revision/config/split preflight and data lock |
| DATA-03 | normalizer golden/adversarial tests and failure taxonomy | spot-check raw prompt/generation/normalized/gold records |
| DATA-04 | row-order-invariant group partition and injected leakage failures | review zero-overlap audit and protected-access manifest |
| DATA-05 | 99/100 sufficiency boundary tests; natural-only counting | per-domain/regime real class report before fitting |
| DATA-06 | corruption provenance round-trip and primary-iterator rejection | verify separate corruption artifact/counts |
| DATA-07 | fake tokenizer/model regime, seed, count, and cap tests | one real generation per regime plus corpus budget report |

### Recommended commands and feedback cadence

- Focused schema/normalizer loop: `uv run pytest tests/test_evidence.py
  tests/test_normalization.py -q`
- Adapter/partition loop: `uv run pytest tests/test_adapters.py
  tests/test_partitions.py -q`
- Generation loop: `uv run pytest tests/test_generation.py -q`
- Every wave: `uv run pytest` and `uv run ruff check .`
- Explicit real checks are never part of pytest/CI and must use a separate output
  root so they cannot mutate unit fixtures.

The phase validation strategy should keep automated feedback under two minutes.
Real preparation and generation are checkpointed manual acceptance work whose
artifacts, not console output, constitute evidence.
