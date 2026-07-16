#!/usr/bin/env python3
"""Resumable Colab/A100 launcher for sealed Phase 2 generation evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, env=os.environ.copy())


def validate_prepared(root: Path, repo: Path) -> None:
    code = (
        "import json; from pathlib import Path; "
        "from seer.runtime import ArtifactRecord,validate_artifacts; "
        f"r=Path({str(root)!r}); assert (r/'COMPLETE').is_file(); "
        "m=json.loads((r/'manifest.json').read_text()); "
        "validate_artifacts(r,[ArtifactRecord(**x) for x in m['artifacts']]); "
        "a=json.loads((r/'leakage-audit.json').read_text()); "
        "assert not a['content_overlaps'] and not a['group_overlaps']; print('prepared corpus OK')"
    )
    run("uv", "run", "--offline", "python", "-c", code, cwd=repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("examples/evidence.json"))
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--run-primary", action="store_true")
    parser.add_argument("--run-thinking", action="store_true")
    args = parser.parse_args()
    repo, prepared = args.repo_dir.resolve(), args.prepared_root.resolve()
    run("uv", "sync", "--frozen", cwd=repo)
    check = (
        "import torch; assert torch.cuda.is_available(), 'CUDA GPU required'; "
        "p=torch.cuda.get_device_properties(0); "
        "assert 'A100' in p.name, f'A100 required, found {p.name}'; "
        "print(p.name,p.total_memory)")
    run("uv", "run", "python", "-c", check, cwd=repo)
    validate_prepared(prepared, repo)
    if args.allow_model_download:
        run("uv", "run", "huggingface-cli", "download", MODEL, "--revision", REVISION,
            cwd=repo)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    base = ["uv", "run", "--offline", "seer", "cache-outputs", "--config",
            str((repo / args.config).resolve()), "--output-root", str(prepared), "--offline"]
    if args.run_primary:
        run(*base, "--resume", cwd=repo)
    if args.run_thinking:
        run(*base, "--thinking-subset-per-domain", "256", "--resume", cwd=repo)
    print(json.dumps({"primary_requested": args.run_primary,
                      "thinking_requested": args.run_thinking, "prepared_root": str(prepared)}))


if __name__ == "__main__":
    main()
