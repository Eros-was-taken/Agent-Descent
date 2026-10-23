# Project summary: image deblurring and inpainting

## Project status

This repository contains two Python restoration pipelines:

- `pgd_codex.py` performs known-kernel image deblurring and asks a fresh
  Codex CLI session for a bounded full-image correction at each iteration.
- `pgd_qwen.py` performs either known-kernel deblurring or masked inpainting
  and asks a locally served Qwen vision-language model through Ollama for a
  bounded correction at each iteration.

Both pipelines use RGB images represented as `float32` arrays with shape
`H x W x 3` and intensities in `[0, 1]`. They use DeepInv's RGB DnCNN model
with the bundled `dncnn_sigma2_lipschitz_color.pth` checkpoint. The checkpoint
is loaded locally and is not downloaded by the restoration scripts.

Each run saves the observation, problem settings, per-iteration arrays and
previews, correction records, and a final result. `pgd_qwen.py` also saves a
deblurring comparison image or an inpainting stop report. Ground truth is
optional in the Qwen runner; in the Codex runner, `--gt` defaults to
`gt.png`. A reference image is for evaluation only and is not sent to either
correction model.

The tests `test_pgd_codex.py` and `test_pgd_qwen.py` exercise the iterative
logic, bounds, persistence, and error paths offline using test doubles. They
do not verify a live Codex or Ollama/Qwen service or guarantee restoration
quality. The repository currently includes prepared observations and prior
run artifacts under `results/`; the recorded Qwen retry log discussed at the
end documents a timeout.

## Notation and image model

Let:

- `y` be the observed RGB image and `x_k` the current estimate at iteration
  `k`.
- `A` be the known blur operator, formed from a normalized point-spread
  function (PSF); `A*` is its true adjoint.
- `D` be the DnCNN denoiser.
- `clip(v, 0, 1)` clip every image channel to the valid intensity range.
- `||v||_2^2` be the sum of squares over pixels and channels (not the mean).
- `||v||_infinity` be the maximum absolute value over all entries.
- `⊙` denote elementwise multiplication, with a 2D mask broadcast over RGB.

Both runners initialize the estimate with the observation: `x_0 = y`.

The default blur PSF is a square, `17 x 17` discrete Gaussian with standard
deviation `sigma = 2.5` pixels. In general its entries are proportional to

```text
h[i,j] = exp(-(i^2 + j^2) / (2 sigma^2))
```

and are normalized so that `sum(h) = 1`. The convolution boundary convention
is selected by `--padding`. The actual linear operator and adjoint are supplied
by DeepInv. The scripts do not estimate an unknown blur: the PSF and boundary
model need to match the observation.

## Deblurring mathematics

The data-fidelity objective is the half squared residual:

```text
f(x) = 1/2 ||A(x) - y||_2^2
```

Its gradient is:

```text
grad f(x) = A*(A(x) - y)
```

The factor `1/2` cancels the factor of two from differentiating the squared
norm. The adjoint is essential; the gradient is not generally just
`A(x) - y`.

Each iteration applies a data-fidelity gradient step, denoises it, and clips
the result:

```text
z_k       = x_k - alpha * A*(A(x_k) - y)
c_k       = clip(D(z_k), 0, 1)
```

The correction model proposes a residual `d_k`. The host scales it if needed
to satisfy the configured per-entry magnitude bound `rho = max_delta`:

```text
s_k       = min(1, rho / ||d_k||_infinity)     (take s_k = 1 if d_k = 0)
dhat_k    = s_k * d_k
x_(k+1)   = clip(c_k + dhat_k, 0, 1)
```

The in-memory floating-point result feeds the next iteration; PNG files are
quantized previews. Intensity clipping can make the actually applied change
smaller than the bounded proposal.

For `pgd_codex.py`, the model writes a finite `H x W x 3` signed residual in
normalized image units. Its prompt requests explicit, manually selected pixel
coordinates and RGB increments; algorithmic image-restoration methods are
prohibited in that correction step. This is a prompt restriction, not a
mathematical guarantee about how the model generated its values.

For Qwen deblurring, the model returns a JSON list of rectangular edits.
Coordinates are relative values from 0 to 1000, and each signed RGB delta is
specified in 8-bit intensity levels. The host converts levels to normalized
increments by dividing by 255, sums overlapping rectangles, and enforces the
same `max_delta` bound before applying the update.

For an image of width `W` and height `H`, each relative bounding-box endpoint
is converted and clipped as follows:

```text
column = clip(round(x_relative * W / 1000), 0, W - 1)
row    = clip(round(y_relative * H / 1000), 0, H - 1)
```

The resulting rectangle includes both endpoint rows and columns. Its RGB
increment is added to each covered pixel as `delta_8bit / 255`; overlapping
rectangles therefore add before masking and bound scaling.

## Inpainting mathematics

The grayscale mask is converted to an unknown-region mask `M`:

```text
M[i,j] = 1  when grayscale mask intensity >= 128 (unknown pixel)
M[i,j] = 0  otherwise                              (known pixel)
K       = 1 - M                                     (known-region mask)
```

White pixels are unknown and eligible for editing; black pixels are known.
The data-fidelity objective compares only known pixels:

```text
f_M(x) = 1/2 ||K ⊙ (x - y)||_2^2
grad f_M(x) = K ⊙ (x - y)
```

The masked gradient step and denoiser candidate are:

```text
z_k = x_k - alpha * (K ⊙ (x_k - y))
v_k = clip(D(z_k), 0, 1)
```

Known pixels are then projected exactly back to the observation before the
model correction:

```text
c_k = K ⊙ y + M ⊙ v_k
```

The Qwen edit rectangles are accumulated into a residual `d_k`; the host
zeroes it outside the unknown mask and applies the magnitude bound:

```text
d_k = M ⊙ sum_of_rectangle_edits
dhat_k = d_k * min(1, max_delta / ||d_k||_infinity)
```

As in deblurring, use scale `1` for a zero residual. After adding and clipping,
known pixels are projected to the observation once more:

```text
x_(k+1) = K ⊙ y + M ⊙ clip(c_k + dhat_k, 0, 1)
```

Thus known pixels remain exactly equal to the input despite denoising or
correction. Qwen's rectangles are converted from inclusive relative bounding
coordinates to image regions, and overlapping edits add together before the
global magnitude bound is applied.

Inpainting can stop before `--iterations` is reached. It checks all three
signals on each iteration:

```text
relative_fidelity_change =
    |f_k - f_(k-1)| / max(|f_(k-1)|, 1e-12)

applied_correction = ||x_(k+1) - c_k||_infinity

unknown_image_change =
    sqrt(mean((x_(k+1) - x_k)^2 over unknown pixels and RGB channels))
```

The fidelity `f_k` is measured on the denoiser candidate before exact known
pixel projection; measuring after projection would make known-region fidelity
zero by construction. A convergence check passes when each of these is at or
below its respective tolerance. All three must pass for `--patience`
consecutive iterations. `--iterations` is still the hard maximum.

## Quality metric

When ground truth `g` is provided, RGB peak signal-to-noise ratio is reported
as:

```text
MSE = mean((x - g)^2)
PSNR(x, g) = -10 log10(MSE) dB
```

For an exact match, PSNR is positive infinity. In inpainting, PSNR can also be
computed over unknown pixels only. Neither model is given the ground truth or
these quality scores. A lower data-fidelity objective alone does not prove a
better reconstruction.

## Running a problem

Install Python dependencies from `requirements.txt`. The Qwen route additionally
requires Ollama to be running and the selected Qwen model to be available
locally. The Codex route requires the Codex CLI to be installed, signed in, and
configured with an image-capable model.

### Qwen with Ollama

Inpainting (the default task; mask is required):

```powershell
python pgd_qwen.py observation.png --mask mask.png --output results --iterations 20 --gt reference.png
```

Deblurring (mask is not used):

```powershell
python pgd_qwen.py blurred.png --task deblurring --output results --iterations 5 --gt gt.png
```

### Codex CLI

```powershell
python pgd_codex.py blurred.png --output results --iterations 20 --alpha 1.0 --max-delta 0.007843
```

To choose a Codex model and reasoning effort:

```powershell
python pgd_codex.py blurred.png --model gpt-5.6-sol --reasoning-effort medium
```

Run `python pgd_qwen.py --help` or `python pgd_codex.py --help` for the
command-line interface exposed by the corresponding script.

## Parameter reference

All paths are interpreted relative to the current working directory unless
given as absolute paths. Numeric image-scale quantities such as `alpha`,
`max_delta`, and the inpainting tolerances operate on normalized intensities
in `[0, 1]`, not 8-bit integer levels. The checkpoint's training noise level
is fixed by the bundled weights; there is no runtime noise-level option.

### Parameters for `pgd_qwen.py`

| Parameter | Type / choices | Default | Meaning and use |
|---|---|---:|---|
| `image` | Path (required positional) | — | RGB observation. |
| `--task` | `inpainting` or `deblurring` | `inpainting` | Selects the masked or known-blur formulation. |
| `--mask` | Path | None | Required for inpainting; grayscale mask must match image dimensions. Intensity `>=128` means unknown. Ignored for deblurring. |
| `--output` | Path | `results` | Parent directory for task/image/timestamped run output. |
| `--gt` | Path | None | Optional evaluation-only RGB reference with matching dimensions. |
| `--weights` | Path | Bundled checkpoint | Local DnCNN state dictionary. |
| `--iterations` | Positive integer | `5` | Maximum iterations; inpainting may stop earlier. |
| `--timeout` | Positive number of seconds | `600` | Timeout for each Ollama correction attempt. |
| `--retries` | Positive integer | `3` | Ollama correction attempts before raising an error. Retries wait 1, then 2, etc. seconds between attempts. |
| `--alpha` | Number | `1.0` | Step size for the data-fidelity gradient; inpainting allows zero, while deblurring requires a positive value. |
| `--max-delta` | Number | `2/255` (`~0.007843`) | Maximum absolute normalized correction per channel/entry after host scaling; must be positive, and deblurring additionally requires it to be at most `1`. |
| `--fidelity-tol` | Nonnegative number | `0.0001` | Inpainting relative fidelity-change stopping tolerance; unused for deblurring. |
| `--correction-tol` | Nonnegative number | `1/255` (`~0.003922`) | Inpainting applied-correction infinity-norm stopping tolerance; unused for deblurring. |
| `--image-tol` | Nonnegative number | `1/255` (`~0.003922`) | Inpainting unknown-region RMS image-change tolerance; unused for deblurring. |
| `--patience` | Positive integer | `2` | Consecutive inpainting iterations required to meet all stopping tolerances; unused for deblurring. |
| `--model` | Ollama model tag | `qwen3-vl:8b` | Local model name, e.g. another installed Qwen3-VL size. |
| `--device` | PyTorch device string | Auto | Defaults to `cuda:0` if available, otherwise `cpu`; can be set explicitly. |
| `--kernel` | Path to `.npy` | None | Optional known 2D blur PSF for deblurring; odd dimensions, nonnegative values and positive sum are required; host normalizes its sum. If supplied, it takes precedence over Gaussian settings. |
| `--kernel-size` | Positive odd integer | `17` | Gaussian PSF width/height used when no explicit PSF is supplied. |
| `--blur-sigma` | Positive number of pixels | `2.5` | Gaussian PSF standard deviation used when no explicit PSF is supplied. |
| `--padding` | `circular`, `reflect`, `replicate` | `circular` | Boundary convention for the deblurring operator. |

Qwen's internal defaults (not separate CLI options) are Ollama endpoint
`http://localhost:11434/api/chat`, context size `16384`, and a 1-second
increasing retry delay. The default model must be downloaded and the Ollama
service must be available.

### Parameters for `pgd_codex.py`

| Parameter | Type / choices | Default | Meaning and use |
|---|---|---:|---|
| `image` | Path (optional positional) | `blurred.png` | Degraded RGB observation. |
| `--gt` | Path | `gt.png` | Ground truth for evaluation only; the script attempts to load this default path if omitted. |
| `--output` | Path | `results` | Parent directory for timestamped run output. |
| `--iterations` | Positive integer | `5` | Number of correction iterations. |
| `--alpha` | Positive number | `1.0` | Step size for the deblurring data-fidelity gradient. |
| `--max-delta` | Number in `(0,1]` | `2/255` (`~0.007843`) | Maximum absolute normalized entry of the Codex residual after scaling. |
| `--model` | Codex model identifier | CLI configuration | Optional override; model must be available and accept image input. |
| `--reasoning-effort` | Model-supported effort string | CLI configuration | Optional effort override, e.g. `medium` or `high`; availability depends on the selected model. |
| `--timeout` | Positive number of seconds | `600` | Maximum duration for each `codex exec` invocation. |
| `--prompt` | Text | Manual-pixel-edit instruction | Additional restoration-task instruction passed into each iteration context. |
| `--kernel` | Path to `.npy` | None | Optional known 2D blur PSF, normalized by the program. Mutually exclusive with explicit `--blur-sigma` in this CLI. |
| `--kernel-size` | Positive odd integer | `17` | Gaussian PSF width/height if no explicit kernel is used. |
| `--blur-sigma` | Positive number of pixels | `2.5` | Gaussian PSF standard deviation if no explicit kernel is used. |
| `--padding` | `circular`, `reflect`, `replicate` | `circular` | Boundary convention for the deblurring operator. |
| `--weights` | Path | Bundled checkpoint | Local DnCNN state dictionary. |
| `--device` | PyTorch device string | `cpu` | Device for DeepInv blur and DnCNN evaluation, e.g. `cuda:0`. |

The Codex CLI must be installed and signed in. The model and reasoning effort
are inherited from its configuration when their options are omitted. This
script runs Codex separately from the interactive chat and has no retry option.

## Why the model has not produced a correction

The strongest direct evidence in the saved run artifacts is
`results/smiling1_deblurred_deblurring/run_2026_10_19_02_40_58/iteration_0001/qwen_retry.log`:
it records `Attempt 1/5: TimeoutError: timed out`. That means the recorded
request did not return a usable model response within that attempt's timeout.
The iteration contains a prompt but no saved `qwen.txt` response. The log
shows only attempt 1 of 5, so it does **not** establish whether later retries
were made or whether they also failed. It also does not identify why the
request timed out. Possible causes such as a cold model load, insufficient
compute/memory, a stalled or unavailable Ollama service, or inference taking
longer than the timeout are possibilities, not conclusions from this log.

There is also a distinction between a failed request and a valid no-op: the
prompts explicitly allow an empty `edits` list when the model judges that no
visual change is justified. Such a response is valid but applies no
correction. The logged timeout, by contrast, is evidence that this particular
attempt did not provide any response at all.

More generally, the model correction is a small visual heuristic rather than
an optimized inverse solver. It sees images and a short history, not the
ground truth, PSNR, or a differentiable objective to optimize. The permitted
change is tightly bounded (default `2/255` per channel), and the prompt asks
for only a few localized edits. In the Codex route, the prompt further
requires hand-authored pixel coordinates and prohibits algorithmic
restoration in the correction step. These limits can result in a small or
empty correction even when the service works. They do not explain the
recorded timeout, which is an inference/service-response failure.

For diagnosis, first verify that Ollama is running and the exact model tag in
`--model` is installed; then try a small direct Ollama vision request and
review the corresponding `qwen_retry.log`. Increase `--timeout` if inference
is simply slow, and use `--retries` to permit more attempts. A timeout needs
to be resolved before interpreting the model's correction quality. For the
Codex route, separately verify the CLI installation, sign-in, configured
image-capable model, and the per-call `--timeout`.
