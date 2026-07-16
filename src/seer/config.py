"""Strict, versioned experiment configuration and JSON wire helpers."""

from __future__ import annotations

import hashlib
import json
import types
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

Precision = Literal["bf16", "fp32"]
NegativeStrategy = Literal["shuffled", "near_error"]
SCHEMA_VERSION = 1


class ConfigError(ValueError):
    """A configuration error whose message identifies its wire field."""


@dataclass(slots=True)
class ModelConfig:
    base_model_name: str
    concept_dim: int = 256
    freeze_base: bool = True
    commit_layer: int = -1
    revision: str | None = None
    cache_dir: Path | None = None
    local_files_only: bool = True


@dataclass(slots=True)
class DatasetConfig:
    name: str
    revision: str
    train_split: str
    calibration_split: str
    test_splits: list[str]


@dataclass(slots=True)
class SeedConfig:
    training: list[int]
    generation: int
    bootstrap: int


@dataclass(slots=True)
class EnergyConfig:
    negative_strategies: list[NegativeStrategy]
    margin: float = 1.0


@dataclass(slots=True)
class OptimConfig:
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-10
    adamw_weight_decay: float = 0.0


@dataclass(slots=True)
class TrainConfig:
    task_loss_weight: float = 1.0
    energy_loss_weight: float = 1.0
    precision: Precision = "bf16"
    dense_supervision: bool = True
    max_steps: int = 1000
    batch_size: int = 8
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass(slots=True)
class EvalConfig:
    train_domain: str
    transfer_domains: list[str] = field(default_factory=list)
    ece_bins: int = 10
    run_shuffle_control: bool = True


@dataclass(slots=True)
class CalibrationConfig:
    method: Literal["isotonic"] = "isotonic"
    ece_bins: int = 10


@dataclass(slots=True)
class RuntimeConfig:
    backend: Literal["synthetic", "real"] = "synthetic"
    offline: bool = True
    checkpoint_interval: int = 100


@dataclass(slots=True)
class OutputConfig:
    root: Path
    run_name: str
    artifact_schema_version: int = 1


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    dataset: DatasetConfig
    seeds: SeedConfig
    energy: EnergyConfig
    output: OutputConfig
    schema_version: int = SCHEMA_VERSION
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig | None = None
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _decode(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        choices = [item for item in args if item is not type(None)]
        if len(choices) == 1:
            return _decode(value, choices[0], path)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{path}: expected one of {args}, got {value!r}")
        return value
    if is_dataclass(annotation):
        return _decode_dataclass(value, annotation, path)
    if annotation is Path:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected path string")
        return Path(value)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected list")
        return [_decode(item, args[0], f"{path}[{index}]") for index, item in enumerate(value)]
    if origin is tuple:
        if not isinstance(value, list | tuple) or len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} values")
        pairs = enumerate(zip(value, args, strict=True))
        return tuple(_decode(item, kind, f"{path}[{index}]") for index, (item, kind) in pairs)
    if annotation is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if annotation in (str, int, bool) and type(value) is not annotation:
        raise ConfigError(f"{path}: expected {annotation.__name__}")
    return value


def _decode_dataclass(value: Any, cls: type[Any], path: str) -> Any:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected object")
    definitions = {item.name: item for item in fields(cls)}
    unknown = value.keys() - definitions.keys()
    if unknown:
        name = sorted(unknown)[0]
        raise ConfigError(f"{path + '.' if path else ''}{name}: unknown field")
    hints = get_type_hints(cls)
    decoded: dict[str, Any] = {}
    for name, definition in definitions.items():
        field_path = f"{path + '.' if path else ''}{name}"
        if name not in value:
            if definition.default is MISSING and definition.default_factory is MISSING:
                raise ConfigError(f"{field_path}: missing required field")
            continue
        decoded[name] = _decode(value[name], hints[name], field_path)
    return cls(**decoded)


def _validate(config: ExperimentConfig) -> None:
    checks = (
        (
            config.schema_version == SCHEMA_VERSION,
            "schema_version",
            f"unsupported version {config.schema_version}",
        ),
        (bool(config.name.strip()), "name", "must not be empty"),
        (config.model.concept_dim > 0, "model.concept_dim", "must be positive"),
        (config.train.max_steps > 0, "train.max_steps", "must be positive"),
        (config.train.batch_size > 0, "train.batch_size", "must be positive"),
        (config.runtime.checkpoint_interval > 0, "runtime.checkpoint_interval", "must be positive"),
        (config.calibration.ece_bins > 1, "calibration.ece_bins", "must exceed one"),
        (
            len(set(config.seeds.training)) == len(config.seeds.training),
            "seeds.training",
            "must be unique",
        ),
    )
    for valid, path, message in checks:
        if not valid:
            raise ConfigError(f"{path}: {message}")


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Decode a strict JSON-compatible mapping into a validated config."""
    config = _decode_dataclass(data, ExperimentConfig, "")
    _validate(config)
    return config


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Return a JSON-compatible representation with paths and tuples normalized."""
    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value

    return normalize(asdict(config))


def canonical_config_json(config: ExperimentConfig) -> str:
    return json.dumps(
        config_to_dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def config_digest(config: ExperimentConfig) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"config {source}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError("config: expected top-level object")
    return config_from_dict(data)


def write_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config_to_dict(config), indent=2) + "\n", encoding="utf-8")
