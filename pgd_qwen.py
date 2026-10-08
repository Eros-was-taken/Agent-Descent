"""Masked inpainting with DnCNN and bounded Qwen3-VL corrections via Ollama."""

import argparse
import base64
import hashlib
import json
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

WEIGHTS = Path(__file__).parent / "weights/dncnn_sigma2_lipschitz_color.pth"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-vl:8b"
NUM_CTX = 16384
TIMEOUT = 600
FIDELITY_TOL = 1e-4
CORRECTION_TOL = 1 / 255
IMAGE_TOL = 1 / 255
PATIENCE = 2


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_unknown_mask(path, image_shape):
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mask.shape != image_shape[:2]:
        raise ValueError("Mask dimensions must match the observation.")
    return mask >= 128


def masked_fidelity(image, observation, known_mask):
    residual = (image - observation) * known_mask[..., None]
    return float(0.5 * np.square(residual.astype(np.float64)).sum())


def psnr(image, reference, region=None):
    difference = image.astype(np.float64) - reference.astype(np.float64)
    if region is not None:
        difference = difference[region]
    if difference.size == 0:
        return None
    mse = float(np.mean(np.square(difference)))
    return float("inf") if mse == 0 else -10 * np.log10(mse)


def _checked_image(image, expected_shape, label):
    image = np.asarray(image, dtype=np.float32)
    if image.shape != expected_shape or not np.isfinite(image).all():
        raise ValueError(f"{label} must be a finite RGB image with shape {expected_shape}.")
    return np.clip(image, 0, 1)


def bounded_edit(reply, image_shape, unknown_mask, max_delta):
    if not isinstance(reply, dict) or not isinstance(reply.get("edits"), list):
        raise ValueError("Qwen response must contain an edits list.")
    height, width, channels = image_shape
    if channels != 3:
        raise ValueError("Images must have three RGB channels.")
    delta = np.zeros(image_shape, dtype=np.float32)
    for edit in reply["edits"]:
        try:
            left, top, right, bottom = (float(value) for value in edit["bbox_2d"])
            rgb = np.asarray(edit["delta"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid Qwen edit: {edit}") from error
        values = np.array([left, top, right, bottom, *rgb.reshape(-1)], dtype=np.float64)
        if rgb.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid Qwen edit: {edit}")
        x1, x2 = sorted((left, right))
        y1, y2 = sorted((top, bottom))
        column1, column2 = (int(np.clip(round(value / 1000 * width), 0, width - 1)) for value in (x1, x2))
        row1, row2 = (int(np.clip(round(value / 1000 * height), 0, height - 1)) for value in (y1, y2))
        delta[row1:row2 + 1, column1:column2 + 1] += rgb / 255.0

    delta *= unknown_mask[..., None]
    peak = float(np.abs(delta).max())
    if peak > max_delta:
        delta *= max_delta / peak
    return delta, peak


def convergence_metrics(fidelity, previous_fidelity, applied_delta, image, previous_image,
                        unknown_mask, fidelity_tol, correction_tol, image_tol):
    correction_linf = float(np.abs(applied_delta).max())
    if previous_fidelity is None or previous_image is None:
        fidelity_change = None
        image_change = None
    else:
        fidelity_change = abs(fidelity - previous_fidelity) / max(abs(previous_fidelity), 1e-12)
        image_difference = (image - previous_image)[unknown_mask]
        image_change = float(np.sqrt(np.mean(np.square(image_difference.astype(np.float64)))))
    fidelity_ok = fidelity_change is not None and fidelity_change <= fidelity_tol
    correction_ok = correction_linf <= correction_tol
    image_ok = image_change is not None and image_change <= image_tol
    return {
        "relative_fidelity_change": fidelity_change,
        "applied_correction_linf": correction_linf,
        "unknown_image_change_rms": image_change,
        "fidelity_converged": fidelity_ok,
        "correction_near_zero": correction_ok,
        "image_stable": image_ok,
        "all_converged": fidelity_ok and correction_ok and image_ok,
    }


def ollama_corrector(observation, current, unknown_mask, step_dir, history,
                     model=MODEL, num_ctx=NUM_CTX, max_delta=2 / 255):
    height, width, _ = current.shape
    current_path = step_dir / "current.png"
    mask_path = step_dir / "unknown_mask.png"
    Image.fromarray(np.rint(current * 255).astype(np.uint8)).save(current_path)
    Image.fromarray((unknown_mask.astype(np.uint8) * 255)).save(mask_path)
    levels = max(1, round(max_delta * 255))
    prompt = f"""You are the correction step of an image-inpainting experiment.
Image 1 is the RGB observation. Image 2 is the CURRENT iterate after a masked data-fidelity
step and DnCNN denoising. Image 3 is the mask: WHITE pixels are unknown and may be edited;
BLACK pixels are observed and must remain unchanged. All images are {width}x{height} pixels.
Choose a few small rectangular edits only inside white mask regions. For each edit return:
- "bbox_2d": [x1, y1, x2, y2] in relative coordinates 0-1000
- "delta": [R, G, B] signed increments in 8-bit levels, each between -{levels} and {levels}
The host restricts edits to the unknown mask and bounds their total magnitude. Do not invent
details. If no edit is justified, return an empty edits list.
Recent iteration history, oldest first: {json.dumps(history)}
Answer only as JSON with keys "edits" and "explanation".
"""
    (step_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    observation_path = step_dir.parent / "observation.png"
    images = [observation_path, current_path, mask_path]
    encoded = [base64.b64encode(path.read_bytes()).decode("ascii") for path in images]
    body = json.dumps({
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"num_ctx": num_ctx},
        "messages": [{"role": "user", "content": prompt, "images": encoded}],
    }).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        answer = json.loads(response.read())["message"]["content"]
    (step_dir / "qwen.txt").write_text(answer, encoding="utf-8")
    try:
        reply = json.loads(answer)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen returned invalid JSON.") from error
    return reply


def run_inpainting(observation, unknown_mask, output, denoiser, corrector, *,
                   iterations=5, alpha=1.0, max_delta=2 / 255,
                   fidelity_tol=FIDELITY_TOL, correction_tol=CORRECTION_TOL,
                   image_tol=IMAGE_TOL, patience=PATIENCE, ground_truth=None,
                   weights_path=None, corrector_name=MODEL):
    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim != 3 or observation.shape[2] != 3 or not np.isfinite(observation).all():
        raise ValueError("Observation must be a finite RGB image.")
    observation = np.clip(observation, 0, 1)
    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    if unknown_mask.shape != observation.shape[:2] or not unknown_mask.any():
        raise ValueError("Unknown mask must match the image and contain at least one unknown pixel.")
    if iterations < 1 or patience < 1 or alpha < 0 or max_delta <= 0:
        raise ValueError("Iterations, patience, and max_delta must be positive; alpha cannot be negative.")
    if min(fidelity_tol, correction_tol, image_tol) < 0:
        raise ValueError("Convergence tolerances cannot be negative.")
    if ground_truth is not None:
        ground_truth = _checked_image(ground_truth, observation.shape, "Ground truth")

    known_mask = ~unknown_mask
    run_dir = Path(output).resolve() / datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(np.rint(observation * 255).astype(np.uint8)).save(run_dir / "observation.png")
    Image.fromarray(unknown_mask.astype(np.uint8) * 255).save(run_dir / "unknown_mask.png")
    np.save(run_dir / "observation.npy", observation, allow_pickle=False)
    if weights_path is not None:
        weights_path = Path(weights_path)
        weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        weights_name = weights_path.name
    else:
        weights_hash = None
        weights_name = None
    problem = {
        "task": "masked inpainting",
        "mask": "white/1 = unknown; black/0 = observed; grayscale threshold = 128",
        "fidelity": "0.5 * sum((known_mask * (x - observation))**2)",
        "fidelity_for_stopping": "DnCNN candidate before exact known-pixel projection",
        "corrector": corrector_name,
        "weights_file": weights_name,
        "weights_sha256": weights_hash,
        "stop": {
            "fidelity_relative_tolerance": fidelity_tol,
            "applied_correction_tolerance": correction_tol,
            "unknown_image_rms_tolerance": image_tol,
            "patience": patience,
        },
    }
    (run_dir / "problem.json").write_text(json.dumps(problem, indent=2), encoding="utf-8")

    current = observation.copy()
    history = []
    previous_fidelity = None
    previous_result = None
    consecutive = 0
    result = current
    stop_reason = "maximum iterations reached"

    for iteration in range(1, iterations + 1):
        step_dir = run_dir / f"iteration_{iteration:04d}"
        step_dir.mkdir()
        gradient = current - alpha * ((current - observation) * known_mask[..., None])
        candidate = _checked_image(denoiser(gradient), observation.shape, "Denoiser output")
        candidate_fidelity = masked_fidelity(candidate, observation, known_mask)
        denoised = np.where(known_mask[..., None], observation, candidate)
        reply = corrector(observation=observation, current=denoised,
                          unknown_mask=unknown_mask, step_dir=step_dir, history=history)
        delta, proposed_peak = bounded_edit(reply, observation.shape, unknown_mask, max_delta)
        result = np.clip(denoised + delta, 0, 1)
        result = np.where(known_mask[..., None], observation, result).astype(np.float32)
        applied = result - denoised
        result_fidelity = masked_fidelity(result, observation, known_mask)
        metrics = convergence_metrics(
            candidate_fidelity, previous_fidelity, applied, result, previous_result,
            unknown_mask, fidelity_tol, correction_tol, image_tol,
        )
        consecutive = consecutive + 1 if metrics["all_converged"] else 0
        should_stop = consecutive >= patience
        if should_stop:
            stop_reason = f"all convergence criteria met for {patience} consecutive iterations"

        for name, value in {
            "gradient": gradient,
            "denoiser_candidate": candidate,
            "current": denoised,
            "bounded_delta": delta,
            "applied_delta": applied,
            "result": result,
        }.items():
            np.save(step_dir / f"{name}.npy", value, allow_pickle=False)
            preview = value if name in ("current", "denoiser_candidate", "result", "gradient") else 0.5 + 0.5 * value / max_delta
            Image.fromarray(np.rint(np.clip(preview, 0, 1) * 255).astype(np.uint8)).save(step_dir / f"{name}.png")

        record = {
            "iteration": iteration,
            "fidelity_before_projection": candidate_fidelity,
            "fidelity_after_projection": result_fidelity,
            "edits": reply["edits"],
            "reason": reply.get("explanation", ""),
            "proposed_delta_linf": proposed_peak,
            "applied_delta_linf": metrics["applied_correction_linf"],
            **metrics,
            "consecutive_converged_iterations": consecutive,
            "stop": should_stop,
        }
        (step_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if ground_truth is not None:
            record["psnr_full"] = psnr(result, ground_truth)
            record["psnr_unknown"] = psnr(result, ground_truth, unknown_mask)
            (step_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        history = (history + [record])[-10:]
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        fidelity_text = "n/a" if metrics["relative_fidelity_change"] is None else f"{metrics['relative_fidelity_change']:.3g}"
        image_text = "n/a" if metrics["unknown_image_change_rms"] is None else f"{metrics['unknown_image_change_rms']:.3g}"
        print(f"Iteration {iteration}/{iterations}: f={candidate_fidelity:.6g}, "
              f"relative_df={fidelity_text}, correction={metrics['applied_correction_linf']:.3g}, "
              f"image_rms={image_text}, stable_count={consecutive}", flush=True)

        current = result
        previous_fidelity = candidate_fidelity
        previous_result = result.copy()
        if should_stop:
            break

    np.save(run_dir / "result.npy", result, allow_pickle=False)
    Image.fromarray(np.rint(result * 255).astype(np.uint8)).save(run_dir / "result.png")
    (run_dir / "stop.json").write_text(json.dumps({
        "iterations_completed": len(history), "reason": stop_reason,
        "converged": stop_reason != "maximum iterations reached",
    }, indent=2), encoding="utf-8")
    return result, run_dir


def build_denoiser(weights_path, device):
    import deepinv as dinv
    import torch

    network = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained=None)
    network.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    network = network.to(device).eval()

    def denoise(image):
        tensor = torch.from_numpy(image).permute(2, 0, 1)[None].to(device)
        with torch.inference_mode():
            output = network(tensor).clamp(0, 1)
        return output[0].permute(1, 2, 0).cpu().numpy()

    return denoise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qwen-assisted masked image inpainting.")
    parser.add_argument("image", help="RGB observation image")
    parser.add_argument("--mask", required=True, help="grayscale mask: white=unknown, black=observed")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--gt", type=Path, help="optional evaluation-only ground truth")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=2 / 255)
    parser.add_argument("--fidelity-tol", type=float, default=FIDELITY_TOL)
    parser.add_argument("--correction-tol", type=float, default=CORRECTION_TOL)
    parser.add_argument("--image-tol", type=float, default=IMAGE_TOL)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import torch

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    observation = load_rgb(args.image)
    unknown_mask = load_unknown_mask(args.mask, observation.shape)
    ground_truth = load_rgb(args.gt) if args.gt else None
    if ground_truth is not None and ground_truth.shape != observation.shape:
        parser.error("Ground truth dimensions must match the observation.")
    denoiser = build_denoiser(args.weights, device)
    corrector = lambda **kwargs: ollama_corrector(
        **kwargs, model=args.model, max_delta=args.max_delta
    )
    _, run_dir = run_inpainting(
        observation, unknown_mask, args.output, denoiser, corrector,
        iterations=args.iterations, alpha=args.alpha, max_delta=args.max_delta,
        fidelity_tol=args.fidelity_tol, correction_tol=args.correction_tol,
        image_tol=args.image_tol, patience=args.patience,
        ground_truth=ground_truth, weights_path=args.weights, corrector_name=args.model,
    )
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
