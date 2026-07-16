from pathlib import Path

import pytest

from seer.cli import COMMANDS, Invocation, build_parser, main
from seer.config import load_config

EXAMPLE = Path(__file__).parents[1] / "examples" / "synthetic.json"


def test_help_lists_the_complete_milestone_surface(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert all(command in output for command in COMMANDS)


def test_parser_produces_typed_auditable_invocation() -> None:
    invocation = build_parser().parse_args(
        ["smoke", "--config", str(EXAMPLE), "--output-root", "elsewhere", "--offline"]
    )
    assert invocation == Invocation(
        command="smoke",
        config=EXAMPLE,
        output_root=Path("elsewhere"),
        resume=False,
        replace=False,
        offline=True,
        allow_download=False,
    )


def test_resume_and_replace_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["smoke", "--config", str(EXAMPLE), "--resume", "--replace"]
        )


def test_synthetic_example_is_complete_and_valid() -> None:
    config = load_config(EXAMPLE)
    assert config.runtime.backend == "synthetic"
    assert config.model.local_files_only is True


def test_injected_smoke_handler_receives_config_without_model_construction() -> None:
    received = []

    def handler(invocation, config):
        received.append((invocation, config))
        return 7

    assert main(["smoke", "--config", str(EXAMPLE)], handlers={"smoke": handler}) == 7
    assert received[0][1].name == "synthetic-smoke"


def test_deferred_command_fails_honestly_without_external_imports(capsys) -> None:
    assert main(["train", "--config", str(EXAMPLE)]) == 2
    assert "not yet implemented" in capsys.readouterr().err


def test_prepare_download_consent_is_scoped_to_prepare_command() -> None:
    invocation = build_parser().parse_args(
        ["prepare-data", "--config", str(EXAMPLE), "--allow-download"]
    )
    assert invocation.allow_download is True
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--config", str(EXAMPLE), "--allow-download"])


def test_prepare_data_handler_receives_explicit_consent() -> None:
    received = []
    def handler(invocation, config):
        received.append((invocation.allow_download, config))
        return 0
    assert main(["prepare-data", "--config", str(EXAMPLE), "--allow-download"],
                handlers={"prepare-data": handler}) == 0
    assert received[0][0] is True
