from pathlib import Path

import pytest

from seer.cache import (
    ModelCacheError,
    format_download_command,
    require_transformers_version,
    resolve_cached_snapshot,
    verify_snapshot_files,
)


REPOSITORY = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def _snapshot(tmp_path: Path, revision: str = REVISION) -> Path:
    snapshot = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    return snapshot


def test_resolver_receives_exact_revision_and_offline_arguments(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    calls: list[dict[str, object]] = []

    def resolver(repository_id: str, **kwargs: object) -> str:
        calls.append({"repository_id": repository_id, **kwargs})
        return str(snapshot)

    resolved = resolve_cached_snapshot(
        REPOSITORY, REVISION, cache_dir=tmp_path, resolver=resolver
    )

    assert calls == [
        {
            "repository_id": REPOSITORY,
            "revision": REVISION,
            "cache_dir": str(tmp_path),
            "local_files_only": True,
        }
    ]
    assert resolved.repository_id == REPOSITORY
    assert resolved.revision == REVISION
    assert resolved.snapshot_path == snapshot.resolve()
    assert set(resolved.metadata_hashes) == {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    }


def test_cache_miss_is_actionable_and_does_not_retry(tmp_path: Path) -> None:
    calls = 0

    def resolver(repository_id: str, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        raise FileNotFoundError("not cached")

    with pytest.raises(ModelCacheError) as error:
        resolve_cached_snapshot(REPOSITORY, REVISION, cache_dir=tmp_path, resolver=resolver)

    message = str(error.value)
    assert calls == 1
    assert REPOSITORY in message
    assert REVISION in message
    assert str(tmp_path) in message
    assert format_download_command(REPOSITORY, REVISION, tmp_path) in message


def test_snapshot_revision_must_match_requested_commit(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, revision="wrong-commit")

    with pytest.raises(ModelCacheError, match="wrong-commit"):
        verify_snapshot_files(snapshot, REPOSITORY, REVISION, tmp_path)


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("config.json", "configuration"),
        ("tokenizer.json", "tokenizer"),
        ("model.safetensors.index.json", "weights"),
    ],
)
def test_snapshot_requires_essential_files(
    tmp_path: Path, missing: str, expected: str
) -> None:
    snapshot = _snapshot(tmp_path)
    (snapshot / missing).unlink()

    with pytest.raises(ModelCacheError, match=expected):
        verify_snapshot_files(snapshot, REPOSITORY, REVISION, tmp_path)


def test_transformers_version_preflight_rejects_old_version() -> None:
    with pytest.raises(ModelCacheError, match=r"4\.51"):
        require_transformers_version("4.50.3")


def test_transformers_version_preflight_accepts_new_version() -> None:
    require_transformers_version("4.51.0")
