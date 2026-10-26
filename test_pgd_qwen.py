"""Offline tests for mask-aware Qwen inpainting and convergence stopping."""

import json
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pgd_qwen import (bounded_edit, convergence_metrics, load_unknown_mask,
                      iteration_temperature, ollama_json_chat, run_deblurring,
                      run_inpainting, validate_temperature_range)


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


def test_dynamic_temperature_schedule_and_validation():
    temperatures = [
        iteration_temperature(iteration, 5, True, (0.9, 0.2))
        for iteration in range(1, 6)
    ]
    np.testing.assert_allclose(temperatures, [0.9, 0.725, 0.55, 0.375, 0.2])
    assert iteration_temperature(1, 5, False, (0.9, 0.2)) is None
    assert iteration_temperature(1, 1, True, (1, 0)) == 1
    for invalid in ((-0.1, 0.1), (0.8, 1.1), (0.2, 0.8), (float("nan"), 0.1)):
        try:
            validate_temperature_range(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid temperature range was accepted: {invalid}")


def test_run_preserves_known_pixels_and_waits_for_patience():
    observation = np.full((2, 2, 3), 0.5, dtype=np.float32)
    unknown = np.array([[True, False], [False, False]])
    calls = []

    def denoiser(image):
        return np.full_like(image, 0.6)

    def corrector(**kwargs):
        calls.append(kwargs.get("temperature"))
        return {"edits": [], "explanation": "No change needed."}

    with tempfile.TemporaryDirectory() as directory:
        result, run_dir = run_inpainting(
            observation, unknown, directory, denoiser, corrector,
            iterations=6, patience=2, dynamic_temperature=True,
            temperature_range=(0.9, 0.2),
        )
        np.testing.assert_array_equal(result[~unknown], observation[~unknown])
        assert len(calls) == 3
        np.testing.assert_allclose(calls, [0.9, 0.9, 0.725])
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
        from PIL import Image

        with Image.open(run_dir / "comparison.png") as comparison:
            assert comparison.size == (4, 60)


def test_ollama_retries_then_returns_valid_correction():
    with tempfile.TemporaryDirectory() as directory:
        step_dir = Path(directory)
        image_path = step_dir / "input.png"
        from PIL import Image

        Image.new("RGB", (1, 1)).save(image_path)
        answer = json.dumps({"edits": [], "explanation": "No edit needed."})
        response = io.BytesIO(json.dumps({"message": {"content": answer}}).encode("utf-8"))
        with patch("pgd_qwen.urllib.request.urlopen", side_effect=[TimeoutError("busy"), response]) as urlopen, \
                patch("pgd_qwen.time.sleep") as sleep:
            reply = ollama_json_chat("prompt", [image_path], step_dir, retries=2)
        assert reply["edits"] == []
        assert "temperature" not in json.loads(urlopen.call_args.args[0].data)["options"]
        sleep.assert_called_once()
        assert "Succeeded on attempt 2/2" in (step_dir / "qwen_retry.log").read_text(encoding="utf-8")


def test_ollama_receives_temperature_and_deblurring_schedule():
    observation = np.full((1, 1, 3), 0.5, dtype=np.float32)
    temperatures = []

    def corrector(**kwargs):
        temperatures.append(kwargs["temperature"])
        return {"edits": []}

    with tempfile.TemporaryDirectory() as directory:
        run_deblurring(
            observation, directory,
            lambda image, observed: (0.0, np.zeros_like(image)),
            lambda image: image, corrector, iterations=5,
            dynamic_temperature=True, temperature_range=(0.9, 0.2),
        )
    np.testing.assert_allclose(temperatures, [0.9, 0.725, 0.55, 0.375, 0.2])

    with tempfile.TemporaryDirectory() as directory:
        step_dir = Path(directory)
        image_path = step_dir / "input.png"
        from PIL import Image

        Image.new("RGB", (1, 1)).save(image_path)
        response = io.BytesIO(json.dumps({
            "message": {"content": json.dumps({"edits": []})}
        }).encode("utf-8"))
        with patch("pgd_qwen.urllib.request.urlopen", return_value=response) as urlopen:
            ollama_json_chat("prompt", [image_path], step_dir, temperature=0.65)
        request_body = json.loads(urlopen.call_args.args[0].data)
        assert request_body["options"]["temperature"] == 0.65


def test_ollama_reports_exhausted_retries():
    with tempfile.TemporaryDirectory() as directory:
        step_dir = Path(directory)
        image_path = step_dir / "input.png"
        from PIL import Image

        Image.new("RGB", (1, 1)).save(image_path)
        with patch("pgd_qwen.urllib.request.urlopen", side_effect=TimeoutError("Ollama busy")), \
                patch("pgd_qwen.time.sleep"):
            try:
                ollama_json_chat("prompt", [image_path], step_dir, retries=2)
            except RuntimeError as error:
                assert "failed after 2 attempts" in str(error)
            else:
                raise AssertionError("Exhausted retries did not fail.")
        log = (step_dir / "qwen_retry.log").read_text(encoding="utf-8")
        assert "Ollama busy" in log
        assert log.count("Attempt") == 2


def test_deblurring_does_not_claim_success_after_correction_failure():
    observation = np.full((2, 2, 3), 0.5, dtype=np.float32)
    calls = []

    def corrector(**kwargs):
        calls.append(kwargs["step_dir"])
        raise TimeoutError("Ollama timed out")

    with tempfile.TemporaryDirectory() as directory:
        try:
            run_deblurring(
                observation, directory, lambda image, observed: (0.0, np.zeros_like(image)),
                lambda image: image, corrector, iterations=3, run_name="sample_deblurring",
            )
        except TimeoutError:
            pass
        else:
            raise AssertionError("A failed correction was silently accepted.")
        assert len(calls) == 1


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
    test_ollama_retries_then_returns_valid_correction()
    test_ollama_reports_exhausted_retries()
    test_deblurring_does_not_claim_success_after_correction_failure()
    test_patience_resets_and_iteration_cap_wins()
    print("Offline Qwen inpainting checks passed.")