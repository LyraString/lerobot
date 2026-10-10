import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from examples.ur5_real import eval_policy_offline
from examples.ur5_real.eval_policy_offline import summarize_errors


def test_rollout_defaults_to_relocated_smolvla_checkpoint():
    config = yaml.safe_load(Path("examples/ur5_real/rollout.yaml").read_text())
    assert config["policy"]["path"] == "./policy/SmolVLA"


@pytest.mark.parametrize(
    "arguments,expected_path",
    [
        ([], Path("policy/SmolVLA")),
        (["--policy-path=custom/checkpoint"], Path("custom/checkpoint")),
    ],
)
def test_offline_eval_checkpoint_default_and_override(monkeypatch, arguments, expected_path):
    monkeypatch.setattr(sys, "argv", ["eval_policy_offline.py", *arguments])
    load_config = MagicMock(side_effect=RuntimeError("Stop before loading checkpoint"))
    monkeypatch.setattr(eval_policy_offline.PreTrainedConfig, "from_pretrained", load_config)

    with pytest.raises(RuntimeError, match="Stop before loading checkpoint"):
        eval_policy_offline.main()

    load_config.assert_called_once_with(expected_path)


def test_chunk_errors_exclude_padded_frames_and_separate_units():
    predictions = np.zeros((2, 6, 14))
    targets = np.ones((2, 6, 14))
    targets[:, :, :12] = 0.2
    states = np.zeros((2, 14))
    predictions[:, :, :12] = 0.1
    predictions[:, :, 12:] = 0.5
    valid = np.ones((2, 6), dtype=bool)
    valid[1, 1:] = False
    predictions[1, 1:] = 100

    result = summarize_errors(predictions, targets, states, valid)

    assert result["joint_mae_rad"] == pytest.approx(0.1)
    assert result["hold_joint_mae_rad"] == pytest.approx(0.2)
    assert result["gripper_mae_fraction"] == pytest.approx([0.5, 0.5])
    assert [row["step"] for row in result["horizons"]] == [0, 1, 5]
    assert [row["samples"] for row in result["horizons"]] == [2, 1, 1]
    assert all(row["joint_max_rad"] == pytest.approx(0.1) for row in result["horizons"])


def test_chunk_errors_skip_horizons_without_valid_targets():
    predictions = np.zeros((1, 50, 14))
    targets = predictions.copy()
    states = np.zeros((1, 14))
    valid = np.zeros((1, 50), dtype=bool)
    valid[:, 0] = True
    predictions[:, 1:] = 100

    result = summarize_errors(predictions, targets, states, valid)

    assert result["joint_mae_rad"] == 0
    assert [row["step"] for row in result["horizons"]] == [0]
