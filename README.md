# SpectraEdge

**SpectraEdge: A Multi-Scale Multi-Object Edge Detection and Frequency-Domain Analysis System**

A two-person CSE 220 project. The website demonstrates actual Python grayscale preparation, manual Gaussian convolution, **manual Sobel gradients and adjustable thresholding**, and manual Fourier analysis. Prewitt, Laplacian, contours, object measurements, noise experiments, multi-scale processing, detector comparison, and webcam processing are **not implemented**.

## Windows setup and startup

Requires Python 3.12+ and Node.js 22.13+. Tested with Python 3.14.5. Run from `D:\SpectraEdge`.

Install once:

```powershell
cd D:\SpectraEdge
python -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
npm.cmd run setup
```

Terminal 1: start Python and keep the terminal open.

```powershell
cd D:\SpectraEdge
.\backend\.venv\Scripts\python.exe -B -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Terminal 2: start the website and keep the terminal open.

```powershell
cd D:\SpectraEdge
npm.cmd run dev
```

Open `http://localhost:3000`. Backend health: `http://127.0.0.1:8000/health`. Interactive API docs: `http://127.0.0.1:8000/docs`. Stop each service with Ctrl+C; restart Python after backend edits.

`Start Website.cmd` starts **only the frontend**; Python must also be running. No deployment is necessary. Existing hosting configuration is retained but unused by local startup.

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
5. Point out the actual **analyzed dimensions**, **processing time**, and completed stages. Status says **edges computed**, while objects remain **unavailable**, not zero.
6. Download a computed PNG from a card/inspector. Export session downloads actual metadata, without image bytes.
7. Change settings while processing or reset: outdated responses are ignored. Stop Python and Process to demonstrate an actionable connection error; restart it and retry.

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

- Selecting an image stays local until Process sends its File to the configured Python address.
- Uploads and results stay in memory. The server saves no images or result history. Reloading discards session images/results; only device preferences/settings persist.
- Limits: 20 MiB per file, 20 million decoded pixels, one image per request. The preview fits within **512 × 512**, preserves aspect ratio, and is never upscaled.
- EXIF orientation is applied, transparency is composited on white, and animated images use frame zero. Nearest-neighbour preview resizing can alias very fine detail; analyzed dimensions are always shown.
- Gaussian and Fourier calculations use the unchanged manual `signal_ops` modules. Sobel reuses manual convolution; thresholding uses raw magnitudes. Only binary decisions and display PNGs become uint8. Both spectra use one combined maximum.
- Browser cancellation prevents stale results but cannot interrupt a Python calculation already running. One calculation runs at a time; overlap receives a retryable busy response.

## Checks

From the project root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests/signal_ops -p "test_*.py" -v
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests/detection -p "test_*.py" -v
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_api.py" -v
npm.cmd run check
npm.cmd run build
```

With Python running, exercise the **actual frontend API adapter over HTTP**:

```powershell
npm.cmd --prefix frontend run test:integration
```

This checks real PNG uploads and errors, but is not a browser interaction test. Use the teacher sequence for file picking, visual comparison, zoom, and downloads.

## Structure and ownership handoff

- `frontend/`: React/TypeScript website, API adapter, shared result/request state, and tests. Owned by the frontend/signal-operations contributor.
- `backend/signal_ops/` and its tests: existing array-only convolution, Gaussian, DFT, and FFT modules. Same owner; no algorithm duplication in the API.
- `backend/detection/sobel.py`, `threshold.py`, and their tests: Sobel/basic threshold milestone completed by the frontend/signal-operations contributor. Array-only imports: `from backend.detection import sobel, threshold_edges`.
- `backend/app.py` and `backend/tests/test_api.py`: extended for this milestone; **hand off to the teammate for remaining API/pipeline development**. Existing dependency files are unchanged.
- Prewitt, Laplacian, contours, object measurements, noise experiments, multi-scale processing, remaining pipeline, and camera work belong to the teammate. Coordinate future API contract changes through the frontend owner; do not edit the frontend independently.

See [backend/README.md](backend/README.md) for the multipart/JSON contract and [frontend/README.md](frontend/README.md) for integration details. This remains one Git repository. No automatic commits, pushes, or deployments.
