"""Offline tests for mask-aware Qwen inpainting and convergence stopping."""

import json
import tempfile
from pathlib import Path

import numpy as np

from pgd_qwen import (bounded_edit, convergence_metrics, load_unknown_mask,
                      run_deblurring, run_inpainting)


def test_mask_loading_and_bounded_edits():
    with tempfile.TemporaryDirectory() as directory:
        mask_path = Path(directory) / "mask.png"
        from PIL import Image

        Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8)).save(mask_path)
        unknown = load_unknown_mask(mask_path, (1, 4, 3))
    np.testing.assert_array_equal(unknown, [[False, False, True, True]])

    unknown = np.array([[False, True], [False, False]])
    reply = {"edits": [{"bbox_2d": [0, 0, 1000, 1000], "delta": [8, -8, 4]}]}
    delta, peak = bounded_edit(reply, (2, 2, 3), unknown, 2 / 255)
    np.testing.assert_array_equal(delta[~unknown], 0)
    assert np.max(np.abs(delta)) <= 2 / 255
    assert peak > 2 / 255


def test_convergence_signals_are_conjunctive():
    mask = np.array([[True]])
    image = np.zeros((1, 1, 3), dtype=np.float32)
    zero = np.zeros_like(image)
    passing = convergence_metrics(1.0, 1.0, zero, image, image, mask, 1e-4, 1 / 255, 1 / 255)
    assert passing["all_converged"]
    assert passing["fidelity_converged"] and passing["correction_near_zero"] and passing["image_stable"]

    fidelity_fails = convergence_metrics(1.1, 1.0, zero, image, image, mask, 1e-4, 1 / 255, 1 / 255)
    correction_fails = convergence_metrics(1.0, 1.0, np.full_like(image, 0.02), image, image,
                                           mask, 1e-4, 1 / 255, 1 / 255)
    image_fails = convergence_metrics(1.0, 1.0, zero, image, image + 0.02, mask,
                                      1e-4, 1 / 255, 1 / 255)
    assert not fidelity_fails["all_converged"] and not fidelity_fails["fidelity_converged"]
    assert not correction_fails["all_converged"] and not correction_fails["correction_near_zero"]
    assert not image_fails["all_converged"] and not image_fails["image_stable"]


def test_run_preserves_known_pixels_and_waits_for_patience():
    observation = np.full((2, 2, 3), 0.5, dtype=np.float32)
    unknown = np.array([[True, False], [False, False]])
    calls = []

    def denoiser(image):
        return np.full_like(image, 0.6)

    def corrector(**kwargs):
        calls.append(kwargs["current"].copy())
        return {"edits": [], "explanation": "No change needed."}

    with tempfile.TemporaryDirectory() as directory:
        result, run_dir = run_inpainting(
            observation, unknown, directory, denoiser, corrector,
            iterations=6, patience=2,
        )
        np.testing.assert_array_equal(result[~unknown], observation[~unknown])
        assert len(calls) == 3
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        assert [record["iteration"] for record in history] == [1, 2, 3]
        assert history[-1]["consecutive_converged_iterations"] == 2
        stop = json.loads((run_dir / "stop.json").read_text(encoding="utf-8"))
        assert stop["converged"]
        for iteration in range(1, 4):
            step = run_dir / f"iteration_{iteration:04d}"
            current = np.load(step / "current.npy", allow_pickle=False)
            saved_result = np.load(step / "result.npy", allow_pickle=False)
            np.testing.assert_array_equal(current[~unknown], observation[~unknown])
            np.testing.assert_array_equal(saved_result[~unknown], observation[~unknown])


def test_malformed_edit_is_rejected():
    try:
        bounded_edit({"edits": [{"bbox_2d": [0, 0], "delta": [1, 2, 3]}]},
                     (2, 2, 3), np.ones((2, 2), dtype=bool), 2 / 255)
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed Qwen edit was accepted.")


def test_deblurring_loop_applies_bounded_qwen_edits():
    observation = np.full((2, 2, 3), 0.5, dtype=np.float32)
    calls = []

    def objective(image, observed):
        return float(np.square(image - observed).sum()), image - observed

    def corrector(**kwargs):
        calls.append(kwargs["current"].copy())
        return {
            "edits": [{"bbox_2d": [0, 0, 1000, 1000], "delta": [8, -8, 4]}],
            "explanation": "Offline test correction.",
        }

    with tempfile.TemporaryDirectory() as directory:
        result, run_dir = run_deblurring(
            observation, directory, objective, lambda image: image, corrector,
            iterations=2, alpha=0.5, max_delta=2 / 255,
        )
        assert len(calls) == 2
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        assert [record["iteration"] for record in history] == [1, 2]
        assert all(record["applied_delta_linf"] <= 2 / 255 + 1e-6 for record in history)
        assert all(record["loss_after_qwen"] >= 0 for record in history)


def test_patience_resets_and_iteration_cap_wins():
    observation = np.full((1, 1, 3), 0.5, dtype=np.float32)
    unknown = np.ones((1, 1), dtype=bool)
    call_count = 0

    def corrector(**kwargs):
        nonlocal call_count
        call_count += 1
        edits = [{"bbox_2d": [0, 0, 1000, 1000], "delta": [2, 0, 0]}] if call_count == 3 else []
        return {"edits": edits}

    with tempfile.TemporaryDirectory() as directory:
        _, run_dir = run_inpainting(
            observation, unknown, directory, lambda image: image,
            corrector, iterations=3, patience=2, max_delta=2 / 255,
            correction_tol=0.001, image_tol=0.001,
        )
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        assert [record["consecutive_converged_iterations"] for record in history] == [0, 1, 0]
        stop = json.loads((run_dir / "stop.json").read_text(encoding="utf-8"))
        assert not stop["converged"]
        assert stop["iterations_completed"] == 3
        assert stop["reason"] == "maximum iterations reached"


if __name__ == "__main__":
    test_mask_loading_and_bounded_edits()
    test_convergence_signals_are_conjunctive()
    test_run_preserves_known_pixels_and_waits_for_patience()
    test_malformed_edit_is_rejected()
    test_deblurring_loop_applies_bounded_qwen_edits()
    test_patience_resets_and_iteration_cap_wins()
    print("Offline Qwen inpainting checks passed.")