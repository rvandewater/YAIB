"""The CLI must reject unknown arguments and honour the train/evaluate command (#241)."""

from pathlib import Path

import pytest
import yaml

from icu_benchmarks.run_utils import parse_args

REQUIRED = ["-d", "demo_data/mortality24/mimic_demo", "-t", "BinaryClassification"]


@pytest.mark.parametrize(
    ("argv", "evaluate"),
    [
        (REQUIRED, False),
        (["train", *REQUIRED], False),
        (["evaluate", *REQUIRED, "--source-dir", "src"], True),
        ([*REQUIRED, "--eval"], True),
        (["train", *REQUIRED, "--eval"], True),
    ],
)
def test_command(argv, evaluate):
    assert parse_args(argv).eval == evaluate


@pytest.mark.parametrize("extra", [["--tunee"], ["predict"], ["--explain-feature"]])
def test_unknown_arguments_are_rejected(extra):
    with pytest.raises(SystemExit):
        parse_args([*REQUIRED, *extra])


@pytest.mark.parametrize("sweep", sorted(Path("experiments").glob("*.yml")), ids=lambda p: p.stem)
def test_sweep_commands_parse(sweep):
    # wandb fills in ${env} and ${program}; every other token must be a known argument
    command = [str(token) for token in yaml.safe_load(sweep.read_text())["command"] if not str(token).startswith("${")]
    parse_args(command)
