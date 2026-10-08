# Deblurring with DnCNN and terminal Codex

`pgd_codex.py` implements this loop, starting with the blurred observation `x = y`:

```python
for k in range(iterations):
    x_gradient = x - alpha * A_adjoint(A(x) - y)
    current = clip(DnCNN(x_gradient), 0, 1)
    delta = codex_step(current, prompt, last_ten_iterations)
    x = clip(current + bounded(delta), 0, 1)
```

The data fidelity is `f(x) = 0.5 * ||A(x) - y||²`, summed over pixels/channels.
DeepInv supplies the blur operator and its true adjoint. `D` is DeepInv's RGB
DnCNN with the downloaded Lipschitz checkpoint. Codex supplies a full-resolution
signed `H x W x 3` residual in `delta.npy`. The host scales oversized residuals
to satisfy `||delta||_infinity <= max_delta`, then adds them to the current image.
Float32 results feed the next iteration; PNGs are previews only.

The correction prompt requires Codex to hand-author explicit pixel coordinates
and signed RGB increments in a saved helper script. Code assembles those edits
into an otherwise zero delta; algorithmic restoration methods are prohibited
for this correction. This is a prompt instruction, not runtime enforcement of
how Codex chose the values. The host's gradient and DnCNN steps are unchanged.

## Terminal execution

Every iteration invokes **`codex exec`**, using your configured CLI model and
existing sign-in. There is no API SDK, API-key setup, alternate correction
backend, or automatic substitute for Codex. A failed CLI call, missing residual,
invalid array, or timeout stops the loop. A zero residual is valid only if Codex
explicitly supplies it.

Each invocation is a fresh terminal Codex session with explicit image/history
context; it is not this desktop chat's live session. The CLI still communicates
with its model service. The [official CLI documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
describes this execution mode and reuse of existing authentication.

## Prepared, not run

The code and an offline check have been written. No tests, restoration runs,
DnCNN inference, or Codex correction calls have been executed. Dependencies
have not been installed as part of this task. Python, NumPy, Pillow, PyTorch,
DeepInv, and a signed-in image-capable Codex CLI are required.

For later use, with a **known Gaussian blur** (sigma measured in pixels):

```powershell
python -m pip install -r requirements.txt
python pgd_codex.py blurred.png --output results --blur-sigma 2.5 --kernel-size 17 --iterations 20 --alpha 1.0 --max-delta 0.007843
```

Or supply the actual nonnegative PSF as an odd-sized 2D NumPy array (normalized
by the program to sum to one):

```powershell
python pgd_codex.py blurred.png --output results --kernel psf.npy --padding circular
```

`--blur-sigma` and `--kernel` are mutually exclusive overrides. With neither,
the script uses a 17x17 Gaussian PSF with sigma 2.5, matching `blurred.png`.
The default image is `./blurred.png`. Each run creates a folder named
`run_YYYY_MM_DD_HH_MM_SS` inside `./results`, using the computer's local time.
`--output` changes only the parent directory; the run name is automatic.
Thus `python pgd_codex.py` needs no arguments for this prepared image.
The code does not estimate an unknown blur. The kernel and padding must match the image
degradation. Padding defaults to `circular`; `reflect` and `replicate` are also
available. Adjust `--alpha` for your operator. Use `--device cuda:0` for GPU
execution, `--model` and `--reasoning-effort` for CLI overrides, and `--prompt` for restoration
instructions. Images are loaded as RGB in `[0,1]`.

For example, choose a model and effort for every Codex step in this run:

```powershell
python pgd_codex.py --iterations 10 --model gpt-5.6-sol --reasoning-effort medium
```

Omitted options inherit your Codex configuration. The selected model must be
available to your account, accept images, and support the requested effort.
The script forwards effort as `--config model_reasoning_effort="medium"`, using
the documented [CLI configuration overrides](https://learn.chatgpt.com/docs/config-file/config-advanced#one-off-overrides-from-the-cli).

The parent directory may already exist and is created if needed. The generated
run directory is printed at startup. A timestamp collision stops the run without
overwriting existing results. Each iteration saves its arrays, prompt, CLI
log, explanation, and record. Failures preserve completed artifacts for
inspection; automatic resume is not implemented.

## DnCNN checkpoint

Downloaded to `weights/dncnn_sigma2_lipschitz_color.pth` (2,683,949 bytes), using
the same checkpoint that DeepInv selects with `pretrained="download_lipschitz"`.
It is loaded locally with `torch.load(..., weights_only=True)` and strict state
matching. A missing checkpoint is an error; there is no runtime download or
random-weight denoising fallback.

[DeepInv's weight documentation](https://deepinv.org/user_guide/reconstruction/pretrained-models.html)
identifies these as Lipschitz-constrained weights trained at noise level `2/255`
for approximate firm nonexpansiveness. [DnCNN's sigma argument](https://deepinv.org/api/stubs/deepinv.models.DnCNN.html)
is unused, so this script does not expose an adjustable noise level.
The extra Codex correction means the whole loop has no claimed convergence
or ground-truth recovery guarantee.

Source: [DeepInv RGB Lipschitz checkpoint](https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_lipschitz_color.pth).

SHA-256:
`807b47af4b851f3142c2d4a1a11aef3355ae7e191b0173e4f701790fcf2264b6`

## Ten-iteration memory

`history.json` contains exactly the most recent ten completed iterations, or
all completed iterations during startup. Every record includes data losses,
correction magnitude, Codex's explanation, and paths to arrays/previews.
`context.json` embeds those previous records before the current correction;
the same context is included in the CLI prompt. Observation/current images
are attached using the documented [image flags](https://learn.chatgpt.com/docs/developer-commands?surface=cli).
Past numerical/image artifacts are accessible by the paths in the history.

Each iteration retains `gradient.npy`, `current.npy`, Codex's raw `delta.npy`,
`bounded_delta.npy`, `applied_delta.npy` (after intensity clipping), and
`result.npy`, plus previews. The archive grows with the iteration count; only
the history window is limited to ten. `problem.json` records the actual kernel,
boundary convention, objective, and checkpoint hash.

`bounded_delta.png` shows the correction added to `current` before intensity
clipping; `applied_delta.png` shows the actual change after clipping. Both are
RGB visualizations with a fixed scale: `0.5 + 0.5 * (correction / max_delta)`.
Per channel, zero is mid-gray, negative increments are darker, and positive
increments are lighter. PNGs are quantized previews; the corresponding `.npy`
files preserve the exact signed values used in the loop.

Codex runs with `workspace-write` rooted at the experiment directory to write
its residual. Its prompt restricts edits to the current correction artifacts.
Ground truth is used only for evaluation: `--gt` defaults to `./gt.png` and must
match the observation's dimensions. Each iteration prints two RGB PSNR values
in dB against this reference alongside `f`: `PSNR PGD` for `current` (after the
gradient, DnCNN, and intensity clipping), and `PSNR after Codex` for the final
image after adding the bounded correction and clipping.
PSNR is `-10 * log10(mean((x - gt)**2))` for normalized images, or infinity for
an exact match. Neither the reference nor PSNR is included in Codex's prompt or
history. Lower data loss alone does not establish improved restoration quality.

`test_pgd_codex.py` is an offline recurrence/history/bounding check saved for
later authorization. Its test double is confined to that check and is never
used by the restoration program.

## Qwen inpainting with Ollama

`pgd_qwen.py` performs masked inpainting using a data-fidelity step, the local
DnCNN checkpoint, and one bounded Qwen correction per iteration. On Windows,
install Ollama from [ollama.com/download](https://ollama.com/download), or run
the official PowerShell installer:

```powershell
irm https://ollama.com/install.ps1 | iex
ollama pull qwen3-vl:8b
ollama run qwen3-vl:8b
```

Qwen3-VL requires Ollama 0.12.7 or newer. The 8B model download is about 6.1 GB;
`qwen3-vl:4b` and `qwen3-vl:2b` are smaller alternatives. If selecting another
variant, pass its matching tag with `--model`. The script connects to Ollama at
`http://localhost:11434`; keep the Ollama app/service running during inference.
Python dependencies remain those in `requirements.txt`. The DnCNN checkpoint
must also exist at `weights/dncnn_sigma2_lipschitz_color.pth` or be supplied
with `--weights`.

Provide an RGB observation and a co-sized grayscale mask. White mask pixels
(intensity >= 128) are unknown and may be inpainted; black pixels are observed
and are restored exactly from the input after denoising and Qwen correction.
Ground truth is optional and used only for evaluation, never passed to Qwen.

```powershell
python pgd_qwen.py observation.png --mask mask.png --output results --iterations 20 --gt reference.png
```

Early stopping requires all three checks to pass for two consecutive
iterations: relative change in the pre-projection masked fidelity, a near-zero
applied Qwen correction, and stable unknown-region image RMS change. Defaults
are `--fidelity-tol 0.0001`, `--correction-tol 0.00392`, and `--image-tol
0.00392`; adjust them for the image and restoration behavior. `--patience`
changes the required consecutive count, while `--iterations` remains a hard
maximum. Each run records per-iteration signals and the final stop reason.

The fidelity used for convergence is measured on the DnCNN candidate before
known pixels are projected back to the observation. Fidelity after projection
is zero by construction, so comparing only projected iterates would provide no
useful convergence signal. Image stability is measured over unknown pixels so
fixed observed pixels do not dilute changes in the region being restored.

Run the offline checks without Ollama or the DnCNN checkpoint:

```powershell
python test_pgd_qwen.py
```
