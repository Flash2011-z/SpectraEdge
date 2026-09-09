"""Optional pretrained portrait segmentation; never imported by signal_ops.

No downloads or optional-library imports happen when this module is imported.
Only an explicit AI extraction loads the cached local model into memory.
"""

from hashlib import file_digest
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from backend.cutout import paint_marks

MODEL_NAME = "birefnet-portrait"
MODEL_PATH = Path(__file__).resolve().parent / ".models" / f"{MODEL_NAME}.onnx"
# Published by rembg 2.0.84's BiRefNetSessionPortrait download implementation.
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx"
MODEL_MD5 = "c3a64a6abf20250d090cd055f12a3b67"
_session = None
_session_lock = Lock()


class AIUnavailable(RuntimeError):
    """Optional AI setup is missing or cannot load; other methods still work."""


def _verified_model_path():
    if not MODEL_PATH.is_file():
        raise AIUnavailable("AI model is not installed. Run python -m backend.prepare_ai on the backend, or choose a non-AI method.")
    with MODEL_PATH.open("rb") as source:
        checksum = file_digest(source, "md5").hexdigest()
    if checksum != MODEL_MD5:
        raise AIUnavailable("AI model verification failed. Run python -m backend.prepare_ai to repair it, or choose a non-AI method.")
    return str(MODEL_PATH)


def _load_session():
    # Reject absent weights before an expensive optional import/first-use JIT.
    model_path = _verified_model_path()
    # Lazy imports keep a base-only installation usable, even without ONNX.
    try:
        import onnxruntime as ort
        from rembg.sessions.birefnet_portrait import BiRefNetSessionPortrait
    except (ImportError, OSError) as error:
        raise AIUnavailable("Optional AI packages are unavailable. Install backend/requirements-ai.txt and prepare the model, or choose a non-AI method.") from error

    class LocalPortraitSession(BiRefNetSessionPortrait):
        @classmethod
        def download_models(cls, *args, **kwargs):
            # Override ONLY the download hook. Keep rembg's actual pretrained
            # model preprocessing/inference; no network path exists at runtime.
            return model_path

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    # Release temporary activation buffers after inference instead of retaining
    # a multi-gigabyte CPU arena while the user returns to manual processing.
    options.enable_cpu_mem_arena = False
    try:
        return LocalPortraitSession(MODEL_NAME, options, providers=["CPUExecutionProvider"])
    except Exception as error:
        raise AIUnavailable("The local AI model could not load. Restart Python or choose a non-AI method; no fallback was run.") from error


def _get_session():
    global _session
    with _session_lock:
        if _session is None:
            _session = _load_session()
        return _session


def _predict_mask(rgba):
    # The model sees the complete prepared photo. A selection box limits output
    # afterward; it is not a prompt to this portrait model.
    photo = Image.fromarray(rgba[:, :, :3])
    predictions = _get_session().predict(photo)
    if len(predictions) != 1:
        raise RuntimeError("The portrait model returned an unexpected number of masks.")
    mask = np.asarray(predictions[0])
    if mask.shape != rgba.shape[:2] or mask.dtype != np.uint8:
        raise RuntimeError("The portrait model returned an incompatible mask.")
    return mask


def ai_cutout_rgba(rgba, rectangle, marks):
    """Return unchanged RGB with predicted soft alpha, mask and visible bounds.

    The cutout API validates RGBA, rectangle and strokes. Brush edits here are
    direct opacity corrections, NOT learned seeds or a hidden watershed pass.
    """
    predicted = _predict_mask(rgba)
    mask = np.zeros(rgba.shape[:2], dtype=np.uint8)
    x, y, width, height = (rectangle[key] for key in ("x", "y", "width", "height"))
    region = np.s_[y:y + height, x:x + width]
    mask[region] = predicted[region]
    paint_marks(mask, marks, foreground_label=255, background_label=0)
    mask[rgba[:, :, 3] == 0] = 0
    cutout = rgba.copy()
    # Alpha is coverage: combine source transparency with predicted coverage.
    # Convert before multiplication to avoid uint8 overflow; do not alter RGB.
    cutout[:, :, 3] = np.rint(rgba[:, :, 3].astype(float) * mask / 255).astype(np.uint8)
    rows, columns = np.nonzero(cutout[:, :, 3])
    if rows.size == 0:
        raise ValueError("AI found no visible foreground in this selection. Use the whole photo, correct the mask, or choose another method.")
    bounds = {"x": int(columns.min()), "y": int(rows.min()),
              "width": int(columns.max() - columns.min() + 1),
              "height": int(rows.max() - rows.min() + 1)}
    return cutout, mask, bounds
