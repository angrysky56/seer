from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from seer.cli import main
from seer.config import config_to_dict, load_config
from seer.smoke import SmokeInterrupted, run_smoke

EXAMPLE = Path(__file__).parents[1] / "examples" / "synthetic.json"


def _scientific_hash(run_dir: Path) -> str:
    payload = json.loads((run_dir / "artifacts" / "results.json").read_text())
    canonical = json.dumps(payload["records"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _only_run(root: Path) -> Path:
    return next(path for path in root.iterdir() if path.is_dir() and ".replaced." not in path.name)


def test_cli_smoke_is_offline_complete_and_noop(tmp_path: Path, monkeypatch) -> None:
    def blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    argv = ["smoke", "--config", str(EXAMPLE), "--output-root", str(tmp_path), "--offline"]
    assert main(argv) == 0
    run_dir = _only_run(tmp_path)
    expected = {
        "config.json",
        "state.json",
        "manifest.json",
        "COMPLETE",
        "artifacts/results.json",
        "artifacts/steps.jsonl",
        "checkpoints/latest.pt",
    }
    assert expected <= {
        path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()
    }
    before = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    assert main(argv) == 0
    assert before == {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}


def test_interrupted_resume_matches_clean_scientific_hash(tmp_path: Path) -> None:
    config = load_config(EXAMPLE)
    clean = config_to_dict(config)
    clean["output"]["root"] = str(tmp_path / "clean")
    clean_path = tmp_path / "clean.json"
    clean_path.write_text(json.dumps(clean))
    assert main(["smoke", "--config", str(clean_path)]) == 0
    clean_hash = _scientific_hash(_only_run(tmp_path / "clean"))

    interrupted = config_to_dict(config)
    interrupted["output"]["root"] = str(tmp_path / "resumed")
    interrupted_path = tmp_path / "resumed.json"
    interrupted_path.write_text(json.dumps(interrupted))
    resumed_config = load_config(interrupted_path)
    with pytest.raises(SmokeInterrupted):
        run_smoke(resumed_config, interrupt_after_checkpoint=2)
    assert run_smoke(resumed_config, resume=True) == "complete"
    assert _scientific_hash(_only_run(tmp_path / "resumed")) == clean_hash


def test_complete_tamper_fails_closed(tmp_path: Path, capsys) -> None:
    argv = ["smoke", "--config", str(EXAMPLE), "--output-root", str(tmp_path)]
    assert main(argv) == 0
    run_dir = _only_run(tmp_path)
    (run_dir / "artifacts" / "results.json").write_text("{}\n")
    assert main(argv) == 2
    assert "mismatch for artifact" in capsys.readouterr().err
