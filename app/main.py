"""RealWaste classification API (FastAPI + ONNX Runtime).

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "model"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
MAX_BATCH = int(os.getenv("MAX_BATCH", "32"))

META = json.loads((MODEL_DIR / "metadata.json").read_text())
CLASSES: list[str] = META["classes"]
SIZE: int = META["img_size"]
MEAN = np.array(META["mean"], dtype=np.float32).reshape(1, 1, 3)
STD = np.array(META["std"], dtype=np.float32).reshape(1, 1, 3)

# GPU memory control (the GPU is shared with the LLM):
#   GPU_MEM_LIMIT_MB  hard cap for ONNX Runtime's CUDA memory arena (0 = no cap)
#   USE_GPU=0         force CPU
#   WARMUP_BATCH      run one dummy batch at startup so memory is allocated (and any OOM shows) immediately
GPU_MEM_LIMIT_MB = int(os.getenv("GPU_MEM_LIMIT_MB", "0"))
USE_GPU = os.getenv("USE_GPU", "1") == "1"
WARMUP_BATCH = int(os.getenv("WARMUP_BATCH", "16"))

_cuda_opts = {
    "arena_extend_strategy": "kSameAsRequested",  # grow only by what is needed, not in powers of two
    "cudnn_conv_algo_search": "HEURISTIC",        # skip exhaustive search and its large scratch buffers
    "cudnn_conv_use_max_workspace": "0",
}
if GPU_MEM_LIMIT_MB > 0:
    _cuda_opts["gpu_mem_limit"] = GPU_MEM_LIMIT_MB * 1024 * 1024

_providers = []
if USE_GPU and "CUDAExecutionProvider" in ort.get_available_providers():
    _providers.append(("CUDAExecutionProvider", _cuda_opts))
_providers.append("CPUExecutionProvider")
SESSION = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=_providers)
INPUT_NAME = SESSION.get_inputs()[0].name
if WARMUP_BATCH > 0:
    SESSION.run(None, {INPUT_NAME: np.zeros((WARMUP_BATCH, 3, SIZE, SIZE), dtype=np.float32)})

app = FastAPI(title="RealWaste classifier", version="1.0")


def read_upload(file: UploadFile) -> bytes:
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{file.filename}: larger than {MAX_BYTES // (1024 * 1024)} MB")
    return data


def preprocess(data: bytes, name: str) -> np.ndarray:
    """Same as the eval transform in train.py: bilinear resize to SIZE x SIZE, scale, normalize."""
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")  # respect phone camera rotation
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"{name}: not a readable image")
    im = im.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    arr = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def predict_batch_array(x: np.ndarray) -> np.ndarray:
    logits = SESSION.run(None, {INPUT_NAME: x.astype(np.float32)})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def top_k(probs: np.ndarray, k: int) -> list[dict]:
    idx = np.argsort(probs)[::-1][: min(k, len(CLASSES))]
    return [{"label": CLASSES[i], "probability": round(float(probs[i]), 4)} for i in idx]


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model"], "providers": SESSION.get_providers(),
            "gpu_mem_limit_mb": GPU_MEM_LIMIT_MB or None, "max_batch": MAX_BATCH, "classes": CLASSES}


@app.post("/predict")
def predict(file: UploadFile = File(...), top: int = Query(3, ge=1, le=20)):
    x = preprocess(read_upload(file), file.filename)[None]
    probs = predict_batch_array(x)[0]
    preds = top_k(probs, top)
    return {"filename": file.filename, "label": preds[0]["label"], "predictions": preds}


@app.post("/predict/batch")
def predict_batch(files: list[UploadFile] = File(...), top: int = Query(3, ge=1, le=20)):
    if len(files) > MAX_BATCH:
        raise HTTPException(400, f"at most {MAX_BATCH} files per request")
    x = np.stack([preprocess(read_upload(f), f.filename) for f in files])
    probs = predict_batch_array(x)
    return {"results": [
        {"filename": f.filename, "label": top_k(p, 1)[0]["label"], "predictions": top_k(p, top)}
        for f, p in zip(files, probs)
    ]}
