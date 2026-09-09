# Optional local AI-assisted cutout

AI is a third, explicitly selected Object Cutout method. **Edge-guided watershed
remains the default.** No upload, page load, manual extraction, or GrabCut call
loads the AI libraries/model. An AI failure never switches methods or returns a
result labelled as another algorithm.

## Local setup

The base `backend/requirements.txt` is unchanged. On machines that want AI,
run these commands from the repository root, using that machine's backend
virtual environment:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-ai.txt
.\backend\.venv\Scripts\python.exe -B -m backend.prepare_ai
```

Optional packages are pinned to rembg 2.0.84 and ONNX Runtime 1.29.0 (CPU).
Their dependency resolution was checked against the existing base pins; current
Windows wheels installed on Python 3.14.5 without changing those pins. This is
actual installation evidence, not a claim that every rembg release supports that
Python version. There is no global Python installation or GPU requirement.

Setup downloads approximately **973 MB** of BiRefNet portrait ONNX weights
from the fixed rembg release URL and checks its upstream MD5 digest. This
checksum detects incomplete/corrupt downloads; the source URL uses HTTPS.
Weights go in `backend/.models/birefnet-portrait.onnx`, ignored by Git. The
download does not contain or upload user photographs. Do not commit weights or
virtual environments. The setup command can be repeated to repair an incomplete
model. It does not construct an inference session.

Restart the backend after installing packages or updating source:

```powershell
.\backend\.venv\Scripts\python.exe -B -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The website remains local at `http://localhost:3000/cutout`. No deployment is
required. In a base-only install, choosing AI returns an actionable 503 error;
manual watershed, GrabCut and Analyze remain usable.

## User workflow

1. Upload a photo and explicitly choose **AI-assisted cutout — optional**.
2. Click **Extract object**. With no existing rectangle, AI uses the whole photo.
   An existing/drawn rectangle limits retained output; **Reset selection** returns
   AI to the whole photo. It is not a prompt to select one person among several.
3. Inspect Cutout or Mask. AI returns a soft grayscale coverage mask, not the
   binary mask used by the other methods. Edge guidance is disabled: no manual
   Sobel calculation happened in this mode.
4. Optional brushes directly correct opacity. Keep restores source alpha at
   painted pixels; Remove makes them transparent. They do not guide/retrain the
   model or flood neighbouring regions. Later strokes win, including Keep
   outside the rectangle; originally transparent source pixels remain transparent.
5. Download the full-frame or trimmed RGBA PNG. Switching methods immediately
   invalidates the result. Late responses cannot restore stale previews/downloads.

The initial AI extraction loads the verified model into memory. Subsequent AI
requests reuse that one session until Python restarts; switching back to manual
does not unload a previously loaded model. CPU threads are limited to four for
inference, with one inter-op thread. Existing cutout admission serializes work.
The CPU memory arena is disabled so temporary activation buffers are released
after inference rather than held while users switch back to manual mode.
Model memory and first-run latency are additional costs; browser cancellation
discards results but cannot interrupt native inference already in progress.

## Calculation and provenance

`backend/ai_cutout.py` lazily imports rembg/ONNX. A small subclass overrides
only rembg's download hook with our verified local path. **There is no runtime
model-download fallback or configurable cloud model.** Its inherited BiRefNet
preprocessing and pretrained prediction are library implementations, not manual
course work. The model internally resamples for inference; its returned mask
must match the prepared image dimensions before we use it.

The model sees full prepared RGB. The selected rectangle and ordered marks are
applied afterward to the mask. The output RGB is copied exactly. Coverage combines
with source transparency as:

```text
output_alpha = round(source_alpha * mask / 255)
```

No recolouring, inpainting, face generation, manual Gaussian/Sobel, FFT, or
GrabCut enters this path. A soft mask does not by itself remove colour spill
from hair edges. All modes still export at the existing maximum 512-pixel
prepared resolution; AI does not recover detail lost by preparation.

Transparency is not secure redaction: RGB under transparent pixels remains in
the full-frame PNG. Trimming removes only pixels outside the visible bounds.

The existing `/cutout/extract` multipart endpoint accepts `method: "ai-assisted"`
with the same exact prepared PNG, image hash, dimensions, rectangle and marks.
Use a full-frame rectangle to request unrestricted output. Sigma, kernel size,
model names, and unknown settings are rejected. Response provenance is:

```json
{
  "method": "ai-assisted",
  "algorithm": "rembg-onnx",
  "parameters_used": { "model": "birefnet-portrait" },
  "seed_mode": null,
  "guidance_image": null,
  "guidance_scale": null
}
```

PNG fields, source identities, crop bounds and measured milliseconds otherwise
follow the existing cutout contract. `mask_image` contains predicted/corrected
coverage in 0–255; `cutout_image` also incorporates original source alpha.
Mask and source alpha are distinct. Missing packages/weights, corrupt weights,
or load errors return 503 without falling back. Empty output returns 422.
Unexpected inference errors return a safe 500 and release the cutout worker.

## Academic scope and sources

Present this as an **optional pretrained-AI extension**, not a manually
implemented edge detector. It is portrait-oriented and not guaranteed to select
arbitrary objects, one specific person, or perfect hair boundaries. Keep the
manual Gaussian/Sobel path available for the actual Signals and Linear Systems
demonstration and explain the distinction to your adviser.

- [rembg library](https://github.com/danielgatis/rembg)
- [Pinned portrait session and weight source](https://github.com/danielgatis/rembg/blob/v2.0.84/rembg/sessions/birefnet_portrait.py)
- [BiRefNet project](https://github.com/ZhengPeng7/BiRefNet)
- [BiRefNet upstream license](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE)

## Verification

The ordinary test suite requires neither model weights nor optional packages;
AI unit tests stub inference to test contracts, isolation and alpha mathematics.
Stub masks are **not** evidence of model accuracy.

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest backend.tests.test_ai_cutout -q
npm.cmd run check
npm.cmd run build
```

An opt-in live integration check can read a locally supplied photo without
copying it into the repository. Start Python with the model prepared, then:

```powershell
$env:SPECTRAEDGE_TEST_AI_PHOTO = 'C:\path\to\portrait.jpg'
npm.cmd --prefix frontend run test:integration
Remove-Item Env:SPECTRAEDGE_TEST_AI_PHOTO
```

The two original photos supplied during development have additional sparse
foreground/background regression checks. These require the running backend and
the exact original local files, not screenshots:

```powershell
$env:SPECTRAEDGE_TEST_AI_PORTRAIT = 'C:\path\to\images.jpg'
$env:SPECTRAEDGE_TEST_AI_COMPLEX = 'C:\path\to\705959204_1696443868160970_8060873554982327466_n.jpg'
.\backend\.venv\Scripts\python.exe -B -m unittest backend.tests.test_ai_photos -q
Remove-Item Env:SPECTRAEDGE_TEST_AI_PORTRAIT, Env:SPECTRAEDGE_TEST_AI_COMPLEX
```

Without these variables, those two tests skip. They do not download test images,
and the actual photos are not tracked in Git.
