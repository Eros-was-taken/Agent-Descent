"""Offline loop check. Written for later use; no real Codex calls."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pgd_codex import checked_array, codex_step, psnr, run_pgd


def test_loop():
    black = np.zeros((2, 3, 3), dtype=np.float32)
    assert np.isposinf(psnr(black, black))
    assert psnr(black, np.ones_like(black)) == 0.0
    assert abs(psnr(black, np.full_like(black, 0.5)) - 6.020599913) < 1e-8
    observed_history_lengths = []

    def objective(x, y):
        return float(0.2 * np.sum(x)), np.full_like(x, 0.2)

    def fake_codex(step_dir, context, **kwargs):
        assert kwargs["model"] == "gpt-5.6-sol"
        assert kwargs["reasoning_effort"] == "medium"
        k = context["iteration"]
        history = context["history"]
        observed_history_lengths.append(len(history))
        assert [r["iteration"] for r in history] == list(range(max(1, k - 10), k))
        current = np.load(step_dir / "current.npy", allow_pickle=False)
        if k > 1:
            previous = np.load(step_dir.parent / history[-1]["result"], allow_pickle=False)
            np.testing.assert_allclose(current, (previous - 0.02) * 0.9, atol=1e-6)
        return np.full_like(current, 0.5), "Oversized residual to check host bounding."

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "results"
        output.mkdir()
        y = np.full((2, 3, 3), 0.5, dtype=np.float32)
        with patch("pgd_codex.codex_step", side_effect=fake_codex):
            result = run_pgd(y, output, iterations=12, alpha=0.1, max_delta=0.01,
                             objective=objective, denoiser=lambda x: x * 0.9,
                             model="gpt-5.6-sol", reasoning_effort="medium")
        runs = list(output.iterdir())
        assert len(runs) == 1 and runs[0].is_dir()
        output = runs[0]
        datetime.strptime(output.name, "run_%Y_%m_%d_%H_%M_%S")
        expected = y.copy()
        for _ in range(12):
            expected = np.clip((expected - 0.02) * 0.9 + 0.01, 0, 1)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        np.testing.assert_array_equal(y, np.full_like(y, 0.5))
        assert observed_history_lengths == [min(k, 10) for k in range(12)]
        history = json.loads((output / "history.json").read_text(encoding="utf-8"))
        assert [r["iteration"] for r in history] == list(range(3, 13))
        assert all(r["applied_delta_linf"] <= 0.010001 for r in history)
        for invalid in (np.zeros((1, 1)), np.full(y.shape, np.nan)):
            try:
                checked_array(invalid, y.shape)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid Codex output was accepted.")


def test_codex_options():
    with tempfile.TemporaryDirectory() as directory:
        step_dir = Path(directory) / "iteration_0001"
        step_dir.mkdir()
        np.save(step_dir / "delta.npy", np.zeros((2, 3, 3), dtype=np.float32))
        (step_dir / "reason.txt").write_text("Offline check.", encoding="utf-8")
        with patch("pgd_codex.shutil.which", return_value="codex"), patch("pgd_codex.subprocess.run") as cli:
            for effort in (None, "medium"):
                codex_step(step_dir, {"shape": [2, 3, 3], "max_delta": 0.01},
                           model="gpt-5.6-sol", reasoning_effort=effort)
                command = cli.call_args.args[0]
                assert command[command.index("--model") + 1] == "gpt-5.6-sol"
                assert ("--config" in command) == (effort is not None)
                if effort is not None:
                    assert command[command.index("--config") + 1] == 'model_reasoning_effort="medium"'


if __name__ == "__main__":
    test_loop()
    test_codex_options()
    print("Offline loop check passed.")
