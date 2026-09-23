# SpectraEdge

**SpectraEdge: A Multi-Scale Multi-Object Edge Detection and Frequency-Domain Analysis System**

A two-person CSE 220 project that has progressed from manual image-signal processing to an interactive analysis website with detector comparison, live webcam analysis, and transparent object extraction. The mathematical core includes manual Gaussian convolution, Sobel/Prewitt gradients, signed Laplacian responses, multi-scale edge persistence, connected-region boundary measurements, Fourier analysis, and reproducible noise experiments. Optional pretrained AI extends portrait cutout while keeping the course's manual calculations available separately.

## Current project progress

| Workspace | Implemented capabilities |
| --- | --- |
| **Analyze** (`/`) | Image preparation, Gaussian smoothing, three edge detectors, adjustable thresholds, spatial and frequency views, connected-region measurements, PNG and metadata exports. |
| **Multi-scale and Noise** (within Analyze) | Edge persistence across selected scales, support-based fusion, and seeded Gaussian or Salt & Pepper noise experiments. |
| **Compare** (`/compare`) | Sobel, Prewitt, and Laplacian outputs from shared preprocessing, with independent thresholds, object counts, and processing times. |
| **Live Signal Analyzer** (`/live`) | Webcam preview and four synchronized processed outputs: grayscale, Gaussian filtered, edge map, and Fourier spectrum. |
| **Live Signal Monitor** (within Live) | Camera status and source resolution, current processing parameters, backend processing time, capture-to-display latency, and approximate processed-output FPS. |
| **Object Cutout** (`/cutout`) | Rectangle selection, optional Keep/Remove brushes, mask inspection, and full-frame or trimmed transparent PNG export using three selectable methods. |

### Object Cutout methods

| Method | Implementation and intended use |
| --- | --- |
| **Edge-guided watershed** (default) | Manual Gaussian/Sobel elevation with scikit-image watershed. Automatic starting regions support extraction without initial brush marks; best with a fairly uniform background. |
| **GrabCut** | OpenCV segmentation as a separate comparison method, using the selection and optional brush corrections. |
| **AI-assisted cutout** (optional) | Local BiRefNet Portrait inference through rembg and ONNX Runtime on CPU. Produces a soft foreground mask while preserving source RGB. Requires separate package and model installation. |

The Live monitor reports processed-output FPS rather than camera capture FPS. Its latency includes capture, request, and display; backend processing time measures server work separately. The Fourier view uses the grayscale input, independently of the edge output.

AI libraries and weights load only when AI extraction is requested. This is a pretrained portrait extension; the project does not train the model. Manual Gaussian, Sobel, and Fourier implementations remain separate from library-backed segmentation.

**[Object Cutout](docs/OBJECT_CUTOUT.md)** at `/cutout` defaults to manual Gaussian/Sobel elevation driving library-backed watershed. Draw a rectangle with a background margin and extract directly: starting regions use the dominant background colour and low-gradient interiors. Best with a fairly uniform background; the object may touch the photo edge. Keep/Remove brushes are optional corrections. Inspect edge guidance and download a transparent PNG. OpenCV GrabCut remains a separate comparison method; FFT is not used for cutout. Photos are prepared at up to 512 pixels per side before selection. Install `backend/requirements.txt` in the existing virtual environment and restart Python after backend changes.

## Windows setup and startup

Requires Python 3.12+ and Node.js 22.13+. Run from your SpectraEdge project directory; replace the example `D:\SpectraEdge` path below with your checkout location.

Install once:

```powershell
cd D:\SpectraEdge
python -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
npm.cmd run setup
```

### Optional AI model installation

After base setup, run these commands from the project root to enable AI-assisted cutout:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-ai.txt
.\backend\.venv\Scripts\python.exe -B -m backend.prepare_ai
```

This installs rembg 2.0.84 and ONNX Runtime 1.29.0, downloads approximately **973 MB** of BiRefNet Portrait weights, and verifies the pinned checksum. The model is stored in `backend/.models/birefnet-portrait.onnx`; weights and the virtual environment are ignored by Git, so each new checkout needs its own setup. Downloading requires internet access; extraction uses the local model and does not send photos to a cloud AI service. No GPU is required.

Restart the backend after installation. First use loads the model and can take substantial time and memory; subsequent requests reuse the session. Analyze, Compare, Live, watershed, and GrabCut work without this optional setup. See [AI-assisted cutout](docs/AI_CUTOUT.md) for details and limitations.

### Start the application

Start both Python and the website with one command and keep the terminal open:

```powershell
cd D:\SpectraEdge
npm.cmd run dev
```

Python reloads automatically after backend edits. Ctrl+C stops both services started by this command. If a healthy SpectraEdge backend is already running on port 8000, it is reused and remains running when this command stops.

For separate terminals, start Python with the following command, then run `npm.cmd run dev:frontend` in another terminal:

```powershell
cd D:\SpectraEdge
.\backend\.venv\Scripts\python.exe -B -m uvicorn backend.app:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
```

Open `http://localhost:3000`. Backend health: `http://127.0.0.1:8000/health`. Interactive API docs: `http://127.0.0.1:8000/docs`.

`Start Website.cmd` also starts both services using `npm.cmd run dev`. No deployment is necessary. Existing hosting configuration is retained but unused by local startup.

### Optional address configuration

The frontend defaults to `http://127.0.0.1:8000`. Copy `frontend/.env.example` to `frontend/.env.local`, edit `NEXT_PUBLIC_API_URL`, and restart the frontend to change it. This is a public address, not a secret.

The backend allows browser origins `http://localhost:3000` and `http://127.0.0.1:3000`. If using another frontend port, set an explicit comma-separated list before starting Python:

```powershell
$env:SPECTRAEDGE_CORS_ORIGINS = "http://localhost:3001,http://127.0.0.1:3001"
```

Keep this demonstration bound to loopback; it is not an authenticated public upload service.

## Teacher demonstration

1. Open Analyze and upload an image containing fine detail. Initially the large panel shows its local original.
2. Keep **Sobel**, sigma **1.2**, and threshold **96**; use the suggested Gaussian kernel and click **Process image**.
3. Inspect the blur, signed Gx/Gy, gradient magnitude, and edge map. Gx is positive toward brighter pixels on the right; Gy is positive toward brighter pixels below. Gray in a derivative image means zero, not a medium-strength edge.
4. Keep the same image, sigma, and kernel. Increase threshold to **200**, then **500**, processing after each change. Weaker edges disappear; no new edge pixels can be added. At **1443**, none remain. The precise pattern depends on the image.
5. Point out the actual **analyzed dimensions**, **processing time**, completed stages, and connected foreground measurements.
6. Download a computed PNG from a card/inspector. Export session downloads actual metadata, without image bytes.
7. Change settings while processing or reset: outdated responses are ignored. Stop Python and Process to demonstrate an actionable connection error; restart it and retry.

Open **Compare** with an uploaded image and select **Compare detectors** to run
Sobel, Prewitt, and Laplacian from one decoded and Gaussian-filtered signal.
Each card reports its binary edge output, connected-object count, detector
processing time, threshold type, and threshold value. Sobel and Prewitt have
independent raw magnitude thresholds; Laplacian has a separate raw
zero-crossing contrast threshold.

For a multi-scale demonstration, enable **Multi-scale**, keep the default scales
0.8, 1.6, and 3.2, and require support from two scales. Each scale starts from
the same grayscale source. The persistence image shows the number of agreeing
scale masks as grayscale, while the fused mask is white where at least two
scales selected the same pixel. Object measurements use that fused mask; the
normal detector cards continue to show the selected single-scale result.

For **Noise**, choose Gaussian (intensity standard deviation) or Salt & Pepper (pixel corruption probability), set strength, and process. Seed 220 makes repeated runs reproducible. The noisy input feeds smoothing, detection, multi-scale, and the input spectrum; the clean grayscale card remains available. Compare uses clean shared preprocessing.

For **Live**, open `/live`, select **Start camera**, and grant camera access on localhost or HTTPS. Each captured frame produces grayscale, Gaussian-filtered, edge, and input-spectrum outputs at up to 256 pixels per side. Detector, sigma, kernel, and threshold are adjustable. Stop releases the webcam; retry is available after processing errors. Frames remain in memory.

Use the **Live Signal Monitor** below the camera preview to explain the source resolution, active parameters, and timing measurements. The source preview runs independently; the four processed outputs belong to the same captured frame. Invalid settings pause requests, and stale results are hidden after settings change.

For **Object Cutout**, upload a photo at `/cutout`, draw a rectangle with a background margin, and use **Extract object** with the default watershed method. Inspect the mask and edge guidance, optionally refine with Keep/Remove, and download a transparent PNG. Switch to GrabCut to compare segmentation methods.

For **AI-assisted cutout**, complete the optional installation, select **AI-assisted cutout — optional**, and click **Extract object**. With no rectangle, the model uses the whole photo; **Reset selection** restores this behavior. A rectangle restricts retained output rather than prompting a particular person. Keep/Remove brushes directly correct opacity. The portrait model may need corrections around hair, multiple people, or unfamiliar objects; exports retain the prepared resolution of at most 512 pixels per side.

The **Illustrative example** is separate, labelled artwork with preset object values. Its Process button is disabled; it is not a calculated detector result.

For a smoothing comparison, set sigma to **0** and process: grayscale/blur and both spectra match. Then set sigma to **2**, use the suggested 13 × 13 kernel, and process again. Compare the Fourier pair on its shared scale. Changing sigma changes the gradient signal too, so hold it fixed when demonstrating threshold alone.

### Mathematics to explain

`convolve2d()` flips each kernel once and uses reflection padding. The convolution kernels are:

```text
Kx = [ 1  0 -1 ]       Ky = [ 1  2  1 ]
     [ 2  0 -2 ]            [ 0  0  0 ]
     [ 1  0 -1 ]            [-1 -2 -1 ]
```

After flipping, Gx is the weighted right-minus-left difference, and Gy is bottom-minus-top. For a unit horizontal ramp, Gx is `(1+2+1) × (right-left) = 4 × 2 = 8` inside the image, while Gy is zero. A vertical 0-to-255 step gives Gx=1020 at the two adjacent columns; reversing the step gives −1020. Reflection makes the outermost derivative zero along the reflected axis.

Magnitude is `sqrt(Gx² + Gy²)`: the strength of change regardless of direction or sign. For Gx=24 and Gy=−32, magnitude is 40. An edge is white (255) only if **raw magnitude > threshold**; equality is black. Thresholding selects pixels, not objects or contours, and does not thin edges.

Both Sobel kernels sum to zero. The Sobel module subtracts one constant brightness offset before convolution, an equivalent operation that prevents floating-point roundoff from turning a flat Gaussian output into edges at threshold zero. It does not clip weak gradients or introduce a tolerance.

Raw derivatives remain signed floats and can exceed 255. Display PNGs are separate: Gx/Gy map −1020…1020 to black…white (zero rounds to gray 128); magnitude maps 0…`1020 × sqrt(2)` to black…white. These fixed scales never affect thresholding. Gaussian smoothing reduces rapid variations before differentiation; both Fourier calculations still use the original grayscale and the same floating-point smoothed array.

## Data and calculation boundaries

- In Analyze, selecting an image stays local until Process sends its File to the configured Python address. Object Cutout uploads on selection to prepare the editable image. Live sends captured frames while the camera is active and processing settings are valid.
- Uploads and results stay in memory. The server saves no images or result history. Reloading discards session images/results; only device preferences/settings persist.
- Limits: 20 MiB per file, 20 million decoded pixels, one image per request. The preview fits within **512 × 512**, preserves aspect ratio, and is never upscaled.
- EXIF orientation is applied and animated images use frame zero. Analyze composites transparency on white; Object Cutout preserves RGBA and combines source alpha with the segmentation mask. Nearest-neighbour preview resizing can alias very fine detail; analyzed dimensions are always shown.
- Gaussian and Fourier calculations use the unchanged manual `signal_ops` modules. Sobel reuses manual convolution; thresholding uses raw magnitudes. Only binary decisions and display PNGs become uint8. Both spectra use one combined maximum.
- Browser cancellation prevents stale results but cannot interrupt a Python calculation already running. Analyze and Compare share one processing slot; Live has its own slot. Overlap within each slot receives a retryable busy response. Live accepts frames up to 2 MiB and fits them within 256 pixels per side.

## Checks

### Latest local AI verification

On **23 September 2026**, the optional AI setup was verified on Windows with Python 3.14.6:

- Pinned AI packages installed in `backend/.venv`; `pip check` reported no broken requirements.
- The 972,666,916-byte BiRefNet Portrait model passed its configured checksum verification.
- All 13 tests in `backend.tests.test_ai_cutout` passed. These use stub inference to check contracts, isolation, and alpha behavior.
- A separate real CPU inference run on a synthetic fixture returned a valid mask and preserved source RGB. Initialization plus inference took approximately 142 seconds on this machine; this is not a general performance benchmark or a portrait-quality evaluation.

### Repeatable checks

From the project root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_*.py" -v
npm.cmd run check
npm.cmd run build
```

With Python running, exercise the **actual frontend API adapter over HTTP**:

```powershell
npm.cmd --prefix frontend run test:integration
npm.cmd --prefix frontend run test:live-integration
```

This checks real PNG uploads and errors, but is not a browser interaction test. Use the teacher sequence for file picking, visual comparison, zoom, and downloads.

To check actual AI extraction through the running backend with your own local photo:

```powershell
$env:SPECTRAEDGE_TEST_AI_PHOTO = 'C:\path\to\portrait.jpg'
npm.cmd --prefix frontend run test:integration
Remove-Item Env:SPECTRAEDGE_TEST_AI_PHOTO
```

The ordinary AI contract tests do not require weights and do not measure segmentation quality. See [AI verification](docs/AI_CUTOUT.md#verification) for the optional photo regression checks.

## Structure and ownership handoff

- `frontend/`: React/TypeScript website, API adapter, shared result/request state, and tests. Owned by the frontend/signal-operations contributor.
- `backend/signal_ops/` and its tests: array-only convolution, Gaussian, reproducible noise, DFT, and FFT modules. Same owner; no algorithm duplication in the API.
- `backend/detection/sobel.py`, `threshold.py`, and their tests: Sobel/basic threshold milestone completed by the frontend/signal-operations contributor. Array-only imports: `from backend.detection import sobel, threshold_edges`.
- `backend/app.py` and `backend/tests/test_api.py`: multipart integration and response contract for detector, object, and optional multi-scale analysis. Existing dependency files are unchanged.
- `backend/detection/prewitt.py` and `laplacian.py`: manual Prewitt, signed Laplacian, and separate zero-crossing decisions, exported by `backend.detection` and selectable through `/analyze`.
- `backend/analysis/components.py` provides manual eight-connected component labeling, outer/hole boundary masks, and object measurements. `backend/analysis/multiscale.py` runs independent Gaussian/detector passes and persistence fusion. `backend/signal_ops/noise.py` supplies seeded Gaussian and Salt & Pepper experiments without changing detector algorithms.
- `backend/live_api.py` handles bounded webcam frames through the same numerical modules. `frontend/components/live-workspace.tsx` and its camera/analysis hooks own the Live lifecycle.
- `backend/analysis/comparison.py` performs one Gaussian pass followed by all three existing detector, edge-decision, and object-analysis paths for `POST /compare`.
- `backend/cutout_api.py` owns image preparation and extraction endpoints. `backend/edge_cutout.py` supplies watershed guidance; `backend/cutout.py` supplies GrabCut and brush operations. The editor is in `frontend/components/cutout-workspace.tsx`.
- `backend/ai_cutout.py` loads and runs the optional local portrait model; `backend/prepare_ai.py` downloads and verifies weights explicitly. Optional packages are pinned in `backend/requirements-ai.txt`.

See [backend/README.md](backend/README.md) for the multipart/JSON contract and [frontend/README.md](frontend/README.md) for integration details. This remains one Git repository. No automatic commits, pushes, or deployments.
