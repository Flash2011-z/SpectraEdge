# SpectraEdge

**SpectraEdge: A Multi-Scale Multi-Object Edge Detection and Frequency-Domain Analysis System**

A two-person CSE 220 project. The website now demonstrates actual Python grayscale preparation, manual Gaussian convolution, and manual Fourier analysis. Detection, contours, noise experiments, detector comparison, and webcam processing are **not implemented**.

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
2. Set sigma to **0**, select a kernel, and click **Process image**. Grayscale and Gaussian-blurred images should match, as should both spectra.
3. Set sigma to **2** and click **Use suggested kernel** (13 × 13). Old results disappear and the workspace asks you to process again.
4. Process again. Expand the blurred image and compare both spectra. Fine detail is reduced; both spectra use the displayed shared scale.
5. Point out the actual **analyzed dimensions**, **processing time**, and completed stages. Detection and object measurements are explicitly **not run**.
6. Download a computed PNG from a card/inspector. Export session downloads actual metadata, without image bytes.
7. Change settings while processing or reset: outdated responses are ignored. Stop Python and Process to demonstrate an actionable connection error; restart it and retry.

The **Illustrative example** is separate, labelled artwork with preset object values. Its Process button is disabled; it is not a calculated detector result.

## Data and calculation boundaries

- Selecting an image stays local until Process sends its File to the configured Python address.
- Uploads and results stay in memory. The server saves no images or result history. Reloading discards session images/results; only device preferences/settings persist.
- Limits: 20 MiB per file, 20 million decoded pixels, one image per request. The preview fits within **512 × 512**, preserves aspect ratio, and is never upscaled.
- EXIF orientation is applied, transparency is composited on white, and animated images use frame zero. Nearest-neighbour preview resizing can alias very fine detail; analyzed dimensions are always shown.
- Gaussian and Fourier calculations use the existing manual `signal_ops` modules. Quantization is confined to PNG encoding. Both spectra use one combined maximum.
- Browser cancellation prevents stale results but cannot interrupt a Python calculation already running. One calculation runs at a time; overlap receives a retryable busy response.

## Checks

From the project root:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests/signal_ops -p "test_*.py" -v
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
- `backend/app.py`, `backend/requirements.txt`, and `backend/tests/test_api.py`: initial API setup created in this milestone; **hand off to the teammate for future API/pipeline development**.
- Future detectors, contours, noise experiments, multi-scale processing, complete pipeline, and camera processing belong to the teammate.

See [backend/README.md](backend/README.md) for the multipart/JSON contract and [frontend/README.md](frontend/README.md) for integration details. This remains one Git repository. No automatic commits, pushes, or deployments.
