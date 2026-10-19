"""Exercise the downloadable-data-free example through its documented CLI."""

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_example(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "examples.train_synthetic", *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("method", ["trpcgrad", "trcagrad"])
@pytest.mark.parametrize("mode", ["adaptive_tr", "fixed_tr", "cosine"])
def test_synthetic_example_trains_through_module_entrypoint(method, mode):
    completed = run_example(
        "--method", method, "--tr-mode", mode, "--steps", "3", "--seed", "11"
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)

    assert report["config"]["method"] == method
    assert report["config"]["tr_mode"] == mode
    assert report["config"]["device"] == "cpu"
    assert report["steps_completed"] == 3
    assert len(report["history"]) == 3
    for state in (report["initial"], report["final"]):
        assert len(state["evaluation_task_losses"]) == 2
        assert all(math.isfinite(loss) for loss in state["evaluation_task_losses"])
        assert len(state["task_radii"]) == 2
        assert all(radius > 0 for radius in state["task_radii"])

    checks = report["checks"]
    assert checks["losses_parameters_and_steps_finite"]
    assert checks["shared_gradients_cleared"]
    assert checks["direct_shared_step_matches_proposal"]
    assert checks["outer_optimizer_preserved_shared_parameters"]
    assert checks["task_heads_updated"]
    assert any(step["shared_step_norm"] > 0 for step in report["history"])
    # Check real training took place without requiring monotonic task losses.
    assert report["initial"]["evaluation_task_losses"] != report["final"]["evaluation_task_losses"]

    if mode == "cosine":
        assert checks["paired_current_matches_gradient_forward"] is None
        assert all(step["paired_shared_step_evaluation"] is None for step in report["history"])
        assert all(0 <= step["cosine_scale"] <= 1 for step in report["history"])
    else:
        assert checks["paired_current_matches_gradient_forward"]
        for step in report["history"]:
            paired = step["paired_shared_step_evaluation"]
            assert len(paired["current_losses"]) == len(paired["candidate_losses"]) == 2
            assert len(paired["ratios"]) == len(paired["valid_ratios"]) == 2
            for ratio, valid in zip(paired["ratios"], paired["valid_ratios"]):
                assert math.isfinite(ratio) if valid else ratio is None

    if mode in {"fixed_tr", "cosine"}:
        assert report["initial"]["task_radii"] == report["final"]["task_radii"]


def test_synthetic_example_repeats_with_same_cpu_seed():
    arguments = ("--steps", "3", "--seed", "19")
    first = run_example(*arguments)
    second = run_example(*arguments)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert json.loads(first.stdout) == json.loads(second.stdout)


def test_synthetic_example_rejects_nonpositive_step_count():
    completed = run_example("--steps", "0")
    assert completed.returncode != 0
    assert "steps must be positive" in completed.stderr
