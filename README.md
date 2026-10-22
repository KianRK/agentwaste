# RealWaste classifier: train on GPU, serve over HTTP

> Local agent with MCP access to this classifier: see [AGENT.md](AGENT.md).

ConvNeXt-Tiny fine-tuned on [RealWaste](https://www.kaggle.com/datasets/joebeachcapital/realwaste)
(9 classes), exported to ONNX and served with FastAPI + ONNX Runtime.
Tuned for Ubuntu 24.04 with an 8 GB RTX 4060.

```
realwaste/
├── train.py               # fine-tune (head, then full), picks best by val macro-F1, test report
├── export_onnx.py         # best.pt -> app/model/model.onnx (+ metadata.json), verified vs PyTorch
├── app/main.py            # FastAPI: /health, /predict, /predict/batch
├── requirements-train.txt
├── requirements-api.txt
├── Dockerfile             # CPU serving image
└── Dockerfile.gpu         # GPU serving image
```

## 1. One-time setup

```bash
# NVIDIA driver (skip if `nvidia-smi` already works)
sudo ubuntu-drivers install && sudo reboot
nvidia-smi                                   # should list the RTX 4060

sudo apt install -y python3-venv
cd realwaste
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# PyTorch with CUDA. Copy the current command from https://pytorch.org/get-started/locally/
# (Linux / Pip / CUDA 12.x). At time of writing:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

pip install -r requirements-train.txt
```

## 2. Download the dataset

Create an API token at kaggle.com → Settings → API, save it to `~/.kaggle/kaggle.json`
(`chmod 600` it), then:

```bash
kaggle datasets download joebeachcapital/realwaste -p data --unzip
```

`train.py` finds the class folders automatically, however deep the zip nests them.

## 3. Train

```bash
python train.py --data-dir data --out-dir runs/convnext_tiny
```

Defaults: 224 px, batch 32, bf16 mixed precision, 2 head-only epochs + 25 full epochs.
That uses roughly 4–5 GB of the 4060's 8 GB and takes about 15–20 minutes.
At the end it prints the test accuracy, macro-F1, a per-class report and a confusion matrix;
everything is also saved in `runs/convnext_tiny/metrics.json`.

Variations worth trying (compare on `metrics.json`):

```bash
# higher resolution, usually +1-2 points; lower the batch to stay inside 8 GB
python train.py --img-size 288 --batch-size 24 --out-dir runs/convnext_tiny_288

# lighter / faster on CPU
python train.py --model efficientnetv2_rw_s.ra2_in1k --out-dir runs/effv2s
python train.py --model mobilenetv3_large_100.ra_in1k --lr 3e-4 --out-dir runs/mnv3
```

If you hit CUDA out-of-memory, halve `--batch-size`. If the GPU sits idle (check `nvidia-smi`),
JPEG decoding is the bottleneck: raise `--workers`.

## 4. Export to ONNX

```bash
python export_onnx.py --run-dir runs/convnext_tiny --out-dir app/model
```

It checks that the ONNX model gives the same predictions as PyTorch and stops if not.

## 5. Serve the API

Without Docker:

```bash
pip install -r requirements-api.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

With Docker (CPU, which is plenty for this model):

```bash
docker build -t realwaste-api .
docker run -d --name realwaste -p 8000:8000 -v "$PWD/app/model:/srv/app/model:ro" realwaste-api
```

With Docker on the GPU (needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
docker build -f Dockerfile.gpu -t realwaste-api:gpu .
docker run -d --name realwaste --gpus all -p 8000:8000 -v "$PWD/app/model:/srv/app/model:ro" realwaste-api:gpu
```

The model is mounted as a volume, so retraining only means re-exporting and `docker restart realwaste`.
`GET /health` shows which execution provider is active (`CUDAExecutionProvider` or `CPUExecutionProvider`).

## 6. Call it

```bash
curl -s localhost:8000/health

curl -s -F "file=@some_bottle.jpg" "localhost:8000/predict?top=3"
# {"filename":"some_bottle.jpg","label":"Plastic",
#  "predictions":[{"label":"Plastic","probability":0.93},{"label":"Glass","probability":0.04},...]}

curl -s -F "files=@a.jpg" -F "files=@b.jpg" localhost:8000/predict/batch
```

Python client:

```python
import requests
with open("some_bottle.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/predict", files={"file": f})
print(r.json()["label"])
```

Interactive docs: http://localhost:8000/docs

Environment variables: `MODEL_DIR` (default `app/model`), `MAX_UPLOAD_MB` (10), `MAX_BATCH` (32).

## Notes

- The API preprocessing (bilinear resize to a square, ImageNet-style normalization) mirrors the
  eval transform in `train.py`. If you change one, change the other.
- RealWaste photos are taken on a fixed rig. Expect lower accuracy on phone photos from the wild;
  collect a few hundred of those to test on, and add them to training if the gap is large.
- Dataset license: CC BY 4.0 (credit the RealWaste authors if you publish results or the model).
