"""Deblurring gradient -> Lipschitz DnCNN -> bounded Codex residual."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def checked_array(value, shape=None):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError("Expected a finite, real numeric image/array.")
    value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Array exceeds float32 range.")
    if shape is not None and value.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {value.shape}.")
    return value


def load_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_image(path, x):
    Image.fromarray(np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)).save(path)


def psnr(x, ground_truth):
    """RGB PSNR in dB for matching images normalized to [0, 1]."""
    mse = np.mean((x.astype(np.float64) - ground_truth.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def deblurring_operators(*, kernel=None, kernel_size=17, blur_sigma=2.5,
                         padding="circular", weights=None, device="cpu"):
    """Build A, its true adjoint, and the locally stored DeepInv DnCNN once."""
    import deepinv as dinv
    import torch

    weights = Path(weights) if weights else Path(__file__).parent / "dncnn_sigma2_lipschitz_color.pth"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing local DnCNN checkpoint: {weights}")
    if padding not in ("circular", "reflect", "replicate"):
        raise ValueError("padding must be circular, reflect, or replicate.")
    if kernel is None:
        if kernel_size < 1 or kernel_size % 2 != 1 or not np.isfinite(blur_sigma) or blur_sigma <= 0:
            raise ValueError("Require an odd positive kernel size and finite blur_sigma > 0.")
        kernel = dinv.physics.functional.gaussian_blur(
            psf_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma)
        ).reshape(kernel_size, kernel_size).numpy()
    kernel = checked_array(kernel)
    if kernel.ndim != 2 or min(kernel.shape) < 1 or any(s % 2 != 1 for s in kernel.shape):
        raise ValueError("Kernel must be a nonempty 2D array with odd dimensions.")
    total = float(kernel.sum(dtype=np.float64))
    if np.any(kernel < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("Blur kernel must be nonnegative with a finite positive sum.")
    kernel = (kernel / total).astype(np.float32)
    physics = dinv.physics.Blur(
        filter=torch.from_numpy(kernel)[None, None].to(device), padding=padding, device=device
    )
    network = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained=None)
    network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    network = network.to(device).eval().requires_grad_(False)

    def tensor(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(device)

    def array(x):
        return x[0].permute(1, 2, 0).cpu().numpy().copy()

    @torch.inference_mode()
    def fidelity(x, y):
        residual = physics.A(tensor(x)) - tensor(y)
        return float(0.5 * residual.double().square().sum()), array(physics.A_adjoint(residual))

    @torch.inference_mode()
    def dncnn(x):
        return array(network(tensor(x)))

    problem = {
        "f": "0.5 * ||A(x) - y||_2^2 (sum, not mean)",
        "gradient": "A_adjoint(A(x) - y)",
        "blur_kernel": kernel.tolist(), "padding": padding,
        "D": "DeepInv DnCNN RGB", "weights_file": weights.name,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    return fidelity, dncnn, problem


def codex_step(step_dir, context, *, model=None, reasoning_effort=None, timeout=600):
    """Ask a fresh CLI session to author a full H x W x 3 signed .npy residual."""
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("Install and sign in to the Codex CLI before running.")
    root = step_dir.parent
    prompt = f"""You are the correction step of an image-restoration experiment.
Inspect the attached images: first the degraded observation, then the CURRENT
post-denoiser iterate. The current image has ALREADY received a data-fidelity
gradient step and DnCNN denoising. Your role is to manually author pixel edits
to the delta using visual judgment and the last ten iterations in the JSON
history. Aim for a meaningful visible improvement within the correction budget.

CONTEXT (also saved in {step_dir.name}/context.json):
{json.dumps(context, indent=2, allow_nan=False)}

Create {step_dir.name}/delta.npy. Any helper scripts or candidate previews must
stay inside that iteration directory. The residual must be a finite float32
NumPy array of exactly
{context['shape']}, HWC RGB, containing SIGNED pixel increments in normalized
image units, NOT a replacement image. Its maximum absolute entry must be at
most {context['max_delta']}. The host computes clip(current + delta, 0, 1).
Inspect the current image, observation, and previous results/corrections;
you may inspect enlarged crops and read individual pixel values. Identify
specific visible defects and decide BY HAND which individual pixels to edit
and the signed R, G, B increments for each. Start with an all-zero delta.
Write an explicit list of (row, column, delta_R, delta_G, delta_B) numeric
literals in a helper script, using zero-based coordinates, one entry per
edited pixel. Every nonzero delta entry must come from that hand-authored list.
Leave unselected pixels at zero. Keep this script in the iteration directory
so the exact chosen coordinates and values can be reviewed.
Do NOT generate corrections using predefined or newly invented mathematical
restoration methods: no gradients, deconvolution, FFTs, sharpening filters,
denoisers, learned models, image-generation services, formula-generated masks,
interpolation, procedural brushes, or automatically optimized strengths.
Choosing a method, its parameters, or a region for an algorithm is NOT manual
pixel editing. Do not calculate a residual and then transcribe it into literals.
Code may only load/inspect images, initialize the zero delta, assign your
explicit edits, validate shape/finiteness/bounds, render previews, and save.
Loops may assign the listed edits; they must not derive additional edits.
These restrictions apply to your correction, not the host's existing PGD step.
Inspect a preview of clip(current + delta, 0, 1), and manually revise the
listed values if needed to control halos, noise, or color shifts. Use the
available magnitude budget where justified by the image. Preserve observed
structure; do not invent details or force edits merely to increase magnitude.
If no pixel edits are justified, explicitly save a zero delta and explain why;
do not substitute an automated correction method.
Ground truth and reference PSNR are unavailable to you; do not search for them.
Do not modify inputs, history, context, or previous outputs. Do not install
packages or run another Codex session. Python with NumPy and Pillow is available
at {sys.executable!r}. Save with np.save(..., allow_pickle=False).
Finish with a short explanation identifying the edited locations, the number
of edited pixels, your chosen signed increments, and their visual purpose.
Describe observed tradeoffs; do not claim an unmeasured loss or PSNR gain.
"""
    (step_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    command = [executable, "exec", "--skip-git-repo-check", "--ephemeral",
               "--sandbox", "workspace-write", "--cd", str(root),
               "--output-last-message", str(step_dir / "reason.txt"),
               "--image", str(root / "observation.png"),
               "--image", str(step_dir / "current.png")]
    if model:
        command.extend(["--model", model])
    if reasoning_effort is not None:
        command.extend(["--config", f"model_reasoning_effort={json.dumps(reasoning_effort)}"])
    command.extend(["--", "-"])
    with (step_dir / "codex.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, input=prompt, text=True, encoding="utf-8",
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    # Missing/invalid output is an error; never silently substitute a fake Codex step.
    delta = np.load(step_dir / "delta.npy", allow_pickle=False)
    reason = (step_dir / "reason.txt").read_text(encoding="utf-8")
    return delta, reason


def run_pgd(y, output, *, objective, denoiser, iterations=20, alpha=1.0,
            max_delta=2 / 255, model=None, reasoning_effort=None, timeout=600, problem=None, ground_truth=None,
            prompt="Manually choose individual pixel coordinates and signed RGB delta values from visual inspection; do not use algorithmic correction methods."):
    """objective(x,y) -> (scalar f, gradient); denoiser(x) -> same-shaped image.

    Arrays are HWC RGB float32. output is the parent for a timestamped run folder.
    The corrected iterate, including its floating-point precision, feeds the
    next gradient step. PNGs are previews only.
    """
    y = checked_array(y)
    if y.ndim != 3 or y.shape[2] != 3 or min(y.shape) < 1:
        raise ValueError("Expected a nonempty H x W x 3 observation.")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("Observation must be normalized to [0, 1].")
    if ground_truth is not None:
        ground_truth = checked_array(ground_truth, y.shape)
        if np.any((ground_truth < 0) | (ground_truth > 1)):
            raise ValueError("Ground truth must be normalized to [0, 1].")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    if not np.isfinite([alpha, max_delta, timeout]).all() or alpha <= 0 or not 0 < max_delta <= 1 or timeout <= 0:
        raise ValueError("Require finite alpha > 0, 0 < max_delta <= 1, timeout > 0.")
    output = Path(output).resolve() / datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {output}", flush=True)
    write_json(output / "problem.json", problem)
    np.save(output / "observation.npy", y, allow_pickle=False)
    save_image(output / "observation.png", y)
    history = []
    write_json(output / "history.json", history)
    x = y.copy()

    for k in range(1, iterations + 1):
        step_dir = output / f"iteration_{k:04d}"
        step_dir.mkdir()
        loss_before, gradient = objective(x.copy(), y.copy())
        x_gradient = checked_array(x - alpha * checked_array(gradient, x.shape), x.shape)
        current = checked_array(denoiser(x_gradient.copy()), x.shape)
        # Project to valid intensities before asking for a bounded residual.
        current = np.clip(current, 0, 1)
        loss_current, _ = objective(current.copy(), y.copy())
        if not np.isfinite([loss_before, loss_current]).all():
            raise ValueError("Objective returned a non-finite loss.")
        np.save(step_dir / "gradient.npy", x_gradient, allow_pickle=False)
        np.save(step_dir / "current.npy", current, allow_pickle=False)
        save_image(step_dir / "current.png", current)
        context = {
            "iteration": k, "shape": list(x.shape), "alpha": alpha,
            "max_delta": max_delta, "task": prompt,
            "problem": problem,
            "observation": "observation.npy",
            "current": f"{step_dir.name}/current.npy",
            "loss_before_gradient": float(loss_before),
            "loss_after_denoiser": float(loss_current),
            "history": history[-10:],
        }
        write_json(step_dir / "context.json", context)
        proposal, reason = codex_step(step_dir, context, model=model,
                                     reasoning_effort=reasoning_effort, timeout=timeout)
        proposal = checked_array(proposal, x.shape)
        # Bound the whole residual by scaling, preserving Codex's proposed direction.
        peak = float(np.max(np.abs(proposal)))
        delta = proposal * (max_delta / peak if peak > max_delta else 1.0)
        x = np.clip(current + delta, 0, 1)
        applied_delta = x - current
        loss_after, _ = objective(x.copy(), y.copy())
        if not np.isfinite(loss_after):
            raise ValueError("Objective returned a non-finite loss after correction.")
        np.save(step_dir / "bounded_delta.npy", delta, allow_pickle=False)
        np.save(step_dir / "applied_delta.npy", applied_delta, allow_pickle=False)
        # Signed RGB previews: -max_delta -> black, zero -> gray, +max_delta -> white.
        save_image(step_dir / "bounded_delta.png", 0.5 + 0.5 * (delta / max_delta))
        save_image(step_dir / "applied_delta.png", 0.5 + 0.5 * (applied_delta / max_delta))
        np.save(step_dir / "result.npy", x, allow_pickle=False)
        save_image(step_dir / "result.png", x)
        record = {
            "iteration": k,
            "current": f"{step_dir.name}/current.npy",
            "result": f"{step_dir.name}/result.npy",
            "preview": f"{step_dir.name}/result.png",
            "delta": f"{step_dir.name}/applied_delta.npy",
            "delta_preview": f"{step_dir.name}/applied_delta.png",
            "delta_preview_encoding": "0.5 + 0.5 * (applied_delta / max_delta), per RGB channel",
            "max_delta": max_delta,
            "loss_before_gradient": float(loss_before),
            "loss_after_denoiser": float(loss_current),
            "loss_after_codex": float(loss_after),
            "proposed_delta_linf": peak,
            "applied_delta_linf": float(np.max(np.abs(applied_delta))),
            "reason": reason,
        }
        write_json(step_dir / "record.json", record)
        history = (history + [record])[-10:]
        write_json(output / "history.json", history)
        psnr_text = (
            f", PSNR PGD = {psnr(current, ground_truth):.2f} dB"
            f", PSNR after Codex = {psnr(x, ground_truth):.2f} dB"
        ) if ground_truth is not None else ""
        print(f"Iteration {k}/{iterations}: f = {loss_after:.6g}{psnr_text}", flush=True)

    np.save(output / "result.npy", x, allow_pickle=False)
    save_image(output / "result.png", x)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", default=Path("blurred.png"), help="Degraded image; loaded as RGB in [0,1] (default: ./blurred.png).")
    parser.add_argument("--gt", type=Path, default=Path("gt.png"), help="Ground truth for PSNR evaluation only (default: ./gt.png).")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Parent directory for timestamped run folders (default: ./results).")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=2 / 255)
    parser.add_argument("--model", help="Optional override of your configured Codex model.")
    parser.add_argument("--reasoning-effort", help="Codex reasoning effort, e.g. medium or high; must be supported by the selected model. Defaults to CLI configuration.")
    parser.add_argument("--timeout", type=float, default=600, help="Seconds per Codex call.")
    parser.add_argument("--prompt", default="Manually choose individual pixel coordinates and signed RGB delta values from visual inspection; do not use algorithmic correction methods.")
    blur = parser.add_mutually_exclusive_group()
    blur.add_argument("--kernel", type=Path, help="Known 2D blur PSF in .npy format; normalized to sum 1.")
    parser.add_argument("--kernel-size", type=int, default=17, help="Gaussian PSF size when --kernel is absent (default: 17).")
    blur.add_argument("--blur-sigma", type=float, default=2.5, help="Known Gaussian blur standard deviation in pixels (default: 2.5).")
    parser.add_argument("--padding", choices=["circular", "reflect", "replicate"], default="circular")
    parser.add_argument("--weights", type=Path, help="Local DnCNN checkpoint; defaults to bundled Lipschitz RGB weights.")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0.")
    args = parser.parse_args()
    y = load_image(args.image)
    ground_truth = load_image(args.gt)
    objective, denoiser, problem = deblurring_operators(
        kernel=np.load(args.kernel, allow_pickle=False) if args.kernel else None,
        kernel_size=args.kernel_size, blur_sigma=args.blur_sigma,
        padding=args.padding, weights=args.weights, device=args.device,
    )
    run_pgd(y, args.output, objective=objective, denoiser=denoiser,
            problem=problem, iterations=args.iterations, ground_truth=ground_truth,
            alpha=args.alpha, max_delta=args.max_delta, model=args.model,
            reasoning_effort=args.reasoning_effort, timeout=args.timeout, prompt=args.prompt)
