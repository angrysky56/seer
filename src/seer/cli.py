"""Auditable command-line surface for SEER experiments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from seer.config import ConfigError, ExperimentConfig, load_config

COMMANDS = (
    "prepare-data",
    "cache-outputs",
    "train",
    "evaluate",
    "run-matrix",
    "build-report",
    "smoke",
)


@dataclass(frozen=True, slots=True)
class Invocation:
    command: str
    config: Path
    output_root: Path | None
    resume: bool
    replace: bool
    offline: bool
    allow_download: bool = False
    babi_archive: Path | None = None
    babi_metadata: Path | None = None
    smoke_per_domain_regime: int | None = None
    thinking_subset_per_domain: int | None = None


Handler = Callable[[Invocation, ExperimentConfig], int]


class _InvocationParser(argparse.ArgumentParser):
    def parse_args(self, args: Sequence[str] | None = None, namespace=None) -> Invocation:
        parsed = super().parse_args(args, namespace)
        return Invocation(
            command=parsed.command,
            config=parsed.config,
            output_root=parsed.output_root,
            resume=parsed.resume,
            replace=parsed.replace,
            offline=parsed.offline,
            allow_download=getattr(parsed, "allow_download", False),
            babi_archive=getattr(parsed, "babi_archive", None),
            babi_metadata=getattr(parsed, "babi_metadata", None),
            smoke_per_domain_regime=getattr(parsed, "smoke_per_domain_regime", None),
            thinking_subset_per_domain=getattr(parsed, "thinking_subset_per_domain", None),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _InvocationParser(prog="seer", description="Run auditable SEER experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--output-root", type=Path)
        policy = child.add_mutually_exclusive_group()
        policy.add_argument("--resume", action="store_true")
        policy.add_argument("--replace", action="store_true")
        child.add_argument("--offline", action="store_true")
        if command == "prepare-data":
            child.add_argument("--allow-download", action="store_true")
            child.add_argument("--babi-archive", type=Path)
            child.add_argument("--babi-metadata", type=Path)
        if command == "cache-outputs":
            child.add_argument("--smoke-per-domain-regime", type=int, choices=(1,))
            child.add_argument("--thinking-subset-per-domain", type=int, choices=(256,))
    return parser


def _effective_config(config: ExperimentConfig, invocation: Invocation) -> ExperimentConfig:
    output = config.output
    runtime = config.runtime
    if invocation.output_root is not None:
        output = replace(output, root=invocation.output_root)
    if invocation.offline:
        runtime = replace(runtime, offline=True)
    return replace(config, output=output, runtime=runtime)


def main(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[str, Handler] | None = None,
) -> int:
    """Parse and dispatch without importing model or dataset implementations."""
    invocation = build_parser().parse_args(argv)
    try:
        config = _effective_config(load_config(invocation.config), invocation)
    except ConfigError as error:
        print(f"seer: {error}", file=sys.stderr)
        return 2

    handler = (handlers or {}).get(invocation.command)
    if handler is not None:
        return handler(invocation, config)
    if invocation.command == "prepare-data":
        from seer.preparation import PreparationError, prepare_data

        try:
            prepare_data(config.datasets, config.output.root,
                         allow_download=invocation.allow_download,
                         babi_archive=invocation.babi_archive,
                         babi_metadata=invocation.babi_metadata)
        except (OSError, PreparationError, ValueError) as error:
            print(f"seer: prepare-data failed: {error}", file=sys.stderr)
            return 2
        return 0
    if invocation.command == "cache-outputs":
        from seer.generation import (
            GenerationError,
            GenerationRegime,
            GenerationRunner,
            cache_outputs,
            load_cached_qwen,
        )

        try:
            tokenizer, model = load_cached_qwen(
                cache_dir=str(config.model.cache_dir) if config.model.cache_dir else None)
            runner = GenerationRunner(
                tokenizer, model, model_id=config.model.base_model_name,
                model_revision=config.model.revision or "",
                tokenizer_revision=config.model.revision or "",
            )
            smoke = invocation.smoke_per_domain_regime
            thinking_subset = invocation.thinking_subset_per_domain
            if smoke and thinking_subset:
                raise ValueError("smoke and thinking-subset selectors are mutually exclusive")
            regimes = (
                tuple(GenerationRegime.primary(domain)
                      for domain in ("gsm8k", "proofwriter", "babi"))
                + (GenerationRegime.thinking(),)
                if smoke else (GenerationRegime.thinking(),) if thinking_subset else None)
            cache_outputs(
                config.output.root,
                config.output.root / "generation-runs" /
                (f"{config.output.run_name}-smoke" if smoke else
                 f"{config.output.run_name}-thinking-{thinking_subset}" if thinking_subset else
                 config.output.run_name),
                runner,
                regimes=regimes,
                resume=invocation.resume,
                replace_existing=invocation.replace,
                smoke_per_domain_regime=smoke,
                thinking_subset_per_domain=thinking_subset,
            )
        except (GenerationError, OSError, RuntimeError, ValueError) as error:
            print(f"seer: cache-outputs failed: {error}", file=sys.stderr)
            return 2
        return 0
    if invocation.command == "smoke":
        from seer.smoke import run_smoke

        try:
            run_smoke(config, resume=invocation.resume, replace=invocation.replace)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"seer: smoke failed: {error}", file=sys.stderr)
            return 2
        return 0
    print(f"seer: {invocation.command} is not yet implemented", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
