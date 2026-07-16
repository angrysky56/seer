# SEER Operations

Phase 1 provides an offline, deterministic synthetic runtime. Real datasets and
confirmatory Qwen3 experiments belong to later phases; tests never access external
weights or data.

## Setup and offline quality checks

Create the locked environment (this may download Python packages, but never model
weights or datasets):

```bash
uv sync
```

Qwen3 requires Transformers 4.51 or newer; the lock currently resolves a compatible
version. Verify the environment and the offline suite with:

```bash
uv lock --check
uv run python -c 'import transformers; print(transformers.__version__)'
uv run pytest
uv run ruff check .
```

Expected: the lock is current, Transformers is at least 4.51, and tests/lint pass.

## Synthetic smoke run

Run the complete CPU path without network access:

```bash
uv run seer smoke --config examples/synthetic.json --output-root /tmp/seer-runs --offline
```

The resulting directory is content-addressed:

```text
/tmp/seer-runs/synthetic-smoke-<config-digest>/
├── config.json
├── state.json
├── manifest.json
├── COMPLETE
├── checkpoints/latest.pt
├── artifacts/results.json
├── artifacts/steps.jsonl
└── logs/
```

Run the same command again to validate every recorded artifact hash and exit without
changing a byte. A missing, truncated, or modified artifact fails closed with a
diagnostic instead of being silently regenerated.

## Resume, replacement, and locks

If a process stops after a checkpoint, resume the same effective configuration:

```bash
uv run seer smoke --config examples/synthetic.json --output-root /tmp/seer-runs --resume --offline
```

Resume rejects a completed run, incompatible config digest, checkpoint schema, or
corrupt state. Ordinary execution also refuses to overwrite an incomplete run.

Replacement is destructive intent made explicit. It atomically renames the previous
directory with a `.replaced.<UTC timestamp>` suffix before creating a fresh run:

```bash
uv run seer smoke --config examples/synthetic.json --output-root /tmp/seer-runs --replace --offline
```

Never remove `.lock` merely because it exists. First confirm the recorded PID and host
no longer own a live writer. Deliberate recovery is exposed by the runtime API and
records a recovered-lock event:

```python
from seer.runtime import RunLock

lock = RunLock("/tmp/seer-runs/<run>/.lock", event_sink=print)
lock.acquire(recover_stale=True)
lock.release()
```

## Artifact interpretation and verification

- `config.json` is the fully effective, versioned input. Its canonical SHA-256 digest
  determines the run directory and compatibility boundary.
- `state.json` records lifecycle and stage state for safe continuation; it is mutable
  until completion and is therefore not self-inventoried.
- `checkpoints/latest.pt` contains model, optimizer, RNG, next-position, and data-order
  state needed to emit every logical training step exactly once.
- `artifacts/results.json` is the versioned result envelope. Its diagnostic scientific
  hash covers only canonical step records, not timestamps or operational metadata.
- `artifacts/steps.jsonl` contains the same scientific records in streaming form.
- `manifest.json` records command, git and environment facts, seeds, model/dataset
  identities, and the byte count and SHA-256 for every immutable artifact.
- `COMPLETE` is written last. Its config digest is the transaction commit marker;
  absence means the run is incomplete regardless of other files.

The supported command-line verifier is the completed-run no-op:

```bash
uv run seer smoke --config examples/synthetic.json --output-root /tmp/seer-runs --offline
```

Expected: exit status zero and no file changes. The lower-level verifier is
`seer.runtime.validate_artifacts`, using the artifact records in `manifest.json`.

## Cached Qwen3 preflight and manual forward check

The primary checkpoint is pinned to
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`. Cache resolution is
local-only and never falls back to the network. Verify its metadata without loading
weights:

```bash
uv run python -c 'from seer.cache import QWEN3_REPOSITORY,QWEN3_REVISION,resolve_cached_snapshot; print(resolve_cached_snapshot(QWEN3_REPOSITORY,QWEN3_REVISION).snapshot_path)'
```

A minimal real-model forward check is manual-only because it needs the developer
cache and substantially more memory than CI:

```bash
uv run python -c 'import torch; from seer.config import ModelConfig; from seer.model import SeerPathAModel; m=SeerPathAModel.from_pretrained(ModelConfig(base_model_name="Qwen/Qwen3-0.6B",revision="c1899de289a04d12100db370d81485cdf75e47ca",concept_dim=32)); print(m(torch.tensor([[1]]))["logits"].shape)'
```

If preflight reports a cache miss, downloading is an explicit, network-capable action:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
```

Review storage, network, and model-license implications before running it. Never make
the cached forward check or download command a unit-test/CI dependency.
