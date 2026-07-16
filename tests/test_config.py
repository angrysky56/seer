import json
from pathlib import Path

import pytest

from seer.config import (
    ConfigError,
    config_digest,
    config_from_dict,
    config_to_dict,
    load_config,
    write_config,
)


def complete_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "synthetic-smoke",
        "model": {
            "base_model_name": "Qwen/Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "cache_dir": ".cache/huggingface",
            "local_files_only": True,
            "concept_dim": 32,
            "freeze_base": True,
            "commit_layer": -1,
        },
        "dataset": {
            "name": "synthetic",
            "revision": "builtin-v1",
            "train_split": "train",
            "calibration_split": "validation",
            "test_splits": ["test"],
        },
        "seeds": {"training": [0, 1, 2], "generation": 11, "bootstrap": 19},
        "energy": {"negative_strategies": ["shuffled", "near_error"], "margin": 1.0},
        "optim": {
            "muon_lr": 0.02,
            "muon_momentum": 0.95,
            "muon_weight_decay": 0.0,
            "adamw_lr": 0.0003,
            "adamw_betas": [0.9, 0.95],
            "adamw_eps": 1e-10,
            "adamw_weight_decay": 0.0,
        },
        "train": {
            "task_loss_weight": 1.0,
            "energy_loss_weight": 1.0,
            "precision": "bf16",
            "dense_supervision": True,
            "max_steps": 4,
            "batch_size": 2,
            "seeds": [0, 1, 2],
        },
        "eval": {
            "train_domain": "synthetic",
            "transfer_domains": [],
            "ece_bins": 10,
            "run_shuffle_control": True,
        },
        "calibration": {"method": "isotonic", "ece_bins": 10},
        "runtime": {"backend": "synthetic", "offline": True, "checkpoint_interval": 2},
        "output": {
            "root": "runs",
            "run_name": "synthetic-smoke",
            "artifact_schema_version": 1,
        },
    }


def test_complete_config_round_trips_and_normalizes_wire_types(tmp_path: Path) -> None:
    config = config_from_dict(complete_config())
    assert config.optim.adamw_betas == (0.9, 0.95)
    assert config.output.root == Path("runs")
    assert config_to_dict(config) == complete_config()

    path = tmp_path / "config.json"
    write_config(config, path)
    assert load_config(path) == config
    assert path.read_text().endswith("\n")


def test_digest_is_canonical_and_stable_across_key_order() -> None:
    data = complete_config()
    reversed_data = dict(reversed(list(data.items())))
    assert config_digest(config_from_dict(data)) == config_digest(config_from_dict(reversed_data))


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda data: data.pop("name"), "name"),
        (lambda data: data["model"].update({"surprise": 1}), "model.surprise"),
        (lambda data: data["runtime"].update({"backend": "cloud"}), "runtime.backend"),
        (lambda data: data["optim"].update({"adamw_betas": [0.9]}), "optim.adamw_betas"),
        (lambda data: data["train"].update({"max_steps": 0}), "train.max_steps"),
        (lambda data: data.update({"schema_version": 99}), "schema_version"),
    ],
)
def test_invalid_configs_name_the_dotted_field(mutate, field: str) -> None:
    data = json.loads(json.dumps(complete_config()))
    mutate(data)
    with pytest.raises(ConfigError, match=field.replace(".", r"\.")):
        config_from_dict(data)


def test_loader_reports_invalid_json_with_file_context(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json")
    with pytest.raises(ConfigError, match="config"):
        load_config(path)
