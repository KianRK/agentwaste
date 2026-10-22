# RealWaste: waste classification with a local AI agent

An end-to-end local project on a single 8 GB GPU:

1. **Train** an image classifier that sorts photos of waste into 9 material classes
   ([RealWaste dataset](https://www.kaggle.com/datasets/joebeachcapital/realwaste)).
2. **Serve** it as an HTTP API (FastAPI + ONNX Runtime on the GPU).
3. **Expose** it to AI agents through an **MCP server**.
4. **Run a local agent** (Qwen3 8B on llama.cpp) that answers questions like
   *"what's in this folder of photos, and which images are you unsure about?"* by calling the classifier.

Everything runs on your own machine; no cloud APIs.

```
 you ─► agent/agent.py ──── OpenAI-compatible HTTP ───► llm          llama.cpp · Qwen3 8B (4-bit)   :8080
        (Python, host)                                                    
              │
              └──────────── MCP (streamable HTTP) ────► mcp          MCP server (Python, SDK v2)     :8100
                                                          │ HTTP
                                                          ▼
                                                        classifier   FastAPI + ONNX Runtime (GPU)   :8000
                                                          ▲
                              train.py ─► export_onnx.py ─┘  (ConvNeXt-Tiny, fine-tuned)
```

Classes: Cardboard · Food Organics · Glass · Metal · Miscellaneous Trash · Paper · Plastic · Textile Trash · Vegetation

## Contents

- [Design choices](#design-choices)
- [Requirements](#requirements)
- [Repository layout](#repository-layout)
- [Part 1: Train and export the classifier](#part-1-train-and-export-the-classifier)
- [Part 2: Run the stack](#part-2-run-the-stack)
- [Part 3: Use the agent](#part-3-use-the-agent)
- [Classifier API](#classifier-api)
- [MCP tools](#mcp-tools)
- [GPU memory budget](#gpu-memory-budget)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Credits and license](#credits-and-license)

## Design choices

| Layer | Choice | Why |
|---|---|---|
| Classifier | ConvNeXt-Tiny (`timm`), ImageNet-22k pretrained, fine-tuned | ~4,750 images is too few to train from scratch; transfer learning reaches the low-to-mid 90s % accuracy |
| Training | two phases (head only, then full network), bf16 mixed precision, class-weighted loss, best checkpoint by validation macro-F1 | the classes are imbalanced, so macro-F1 is the honest metric |
| Serving | ONNX export + ONNX Runtime (CUDA) behind FastAPI | small runtime, no PyTorch at inference, a few ms per image on GPU |
| LLM runtime | llama.cpp `llama-server` | full control over GPU layers, context size and KV-cache precision; OpenAI-compatible API |
| LLM | Qwen3 8B, GGUF `Q4_K_M` (4-bit) | the most reliable tool calling that fits in 8 GB next to the classifier |
| Tool interface | MCP server in its own container (MCP Python SDK v2, streamable HTTP) | any MCP client can use it, not just this agent; it can only read one mounted image folder |
| Agent | ~200 lines of plain Python, no framework | every request, tool call and token count is visible and adjustable |

## Requirements

- Linux (developed on Ubuntu 24.04) with an NVIDIA GPU, 8 GB VRAM, 16 GB RAM
- NVIDIA driver with CUDA 13 support (`nvidia-smi` shows "CUDA Version: 13.x")
- Docker Engine + Docker Compose and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Python 3.12 for training and the agent
- A [Kaggle](https://www.kaggle.com) account (API token) to download the dataset

Check that Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

## Repository layout

```
├── train.py               # fine-tune the classifier, report test metrics
├── export_onnx.py         # checkpoint -> app/model/model.onnx (+ metadata.json), verified against PyTorch
├── app/main.py            # classifier API: /health, /predict, /predict/batch
├── Dockerfile.gpu         # classifier image (CUDA 13 + onnxruntime-gpu)
├── Dockerfile             # classifier image for CPU-only machines
├── mcp_server/            # MCP server: tools that wrap the classifier API
├── agent/agent.py         # the agent loop (llama-server + MCP client)
├── compose.yaml           # llm + classifier + mcp
├── .env.example           # all settings for the stack
├── requirements-train.txt
├── requirements-api.txt
└── AGENT.md               # deeper notes on the agent stack
```

Not in the repository (see `.gitignore`): the dataset (`data/`), checkpoints (`runs/`),
the exported model (`app/model/`) and the LLM weights (`models/`). They are recreated by the steps below.

## Part 1: Train and export the classifier

### Set up the training environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# PyTorch with CUDA (current command: https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Download the dataset

Put your Kaggle API token in `~/.kaggle/kaggle.json` (`chmod 600`), then:

```bash
kaggle datasets download joebeachcapital/realwaste -p data --unzip
```

### Train

```bash
python train.py --data-dir data --out-dir runs/convnext_tiny
```

About 15–20 minutes on an RTX 4060, using 4–5 GB of VRAM. The script:

- makes a stratified 70 / 15 / 15 train / validation / test split (saved to `splits.json`)
- trains the classifier head for 2 epochs, then the whole network for 25 epochs (warmup + cosine LR)
- keeps the checkpoint with the best validation macro-F1 (`best.pt`)
- evaluates once on the test split and prints accuracy, macro-F1, a per-class report and a confusion matrix
  (also saved in `metrics.json`)

Useful options: `--img-size 288 --batch-size 24` (usually 1–2 points better),
`--model <any timm model>`, `--epochs`, `--lr`. Halve `--batch-size` on CUDA out-of-memory.

### Results

<!-- Fill in from runs/convnext_tiny/metrics.json -->

| Model | Input | Test accuracy | Test macro-F1 |
|---|---|---|---|
| ConvNeXt-Tiny | 224 px | – | – |

### Export to ONNX

```bash
python export_onnx.py --run-dir runs/convnext_tiny --out-dir app/model
deactivate
```

This writes `app/model/model.onnx` and `metadata.json` (class names, input size, normalization)
and fails if the ONNX output doesn't match PyTorch.

## Part 2: Run the stack

```bash
cp .env.example .env        # set IMAGE_DIR to the folder with photos the agent may read
mkdir -p models
docker compose up -d --build
docker compose logs -f llm  # first start downloads the model (~5 GB); wait for "server is listening"
docker compose ps           # all three services should be running, classifier "healthy"
```

| Service | Port | Check |
|---|---|---|
| `llm` (llama-server) | 8080 | `curl localhost:8080/health`, built-in chat page at http://localhost:8080 |
| `classifier` | 8000 | `curl localhost:8000/health`: `providers` should start with `CUDAExecutionProvider` |
| `mcp` | 8100 | endpoint `http://localhost:8100/mcp` |

All ports are bound to `127.0.0.1`, so nothing is reachable from other machines.

After retraining: export again, then `docker compose restart classifier` (the model folder is mounted, no rebuild needed).

## Part 3: Use the agent

```bash
python3 -m venv agent/.venv
agent/.venv/bin/pip install -r agent/requirements.txt

agent/.venv/bin/python agent/agent.py                               # interactive chat
agent/.venv/bin/python agent/agent.py "summarise the folder bin_3"   # one task, then exit (scriptable)
agent/.venv/bin/python agent/agent.py -v                            # show tool results, tokens, tok/s
```

Example session (illustrative output):

```
you> Classify everything in bin_3 and tell me which images you're unsure about.
→ classify_folder({"path": "bin_3"})
agent> bin_3 has 26 images: 8 Plastic, 6 Metal, 5 Paper, 4 Cardboard, 3 Glass.
       Unsure about 2: bin_3/img_07.jpg (Plastic 54%, maybe Glass) and bin_3/img_19.jpg (Paper 58%, maybe Cardboard).
```

Interactive commands: `/tools`, `/reset` (clear history), `/exit`.

| Option | Default | Purpose |
|---|---|---|
| `--think` | off | enable Qwen3 reasoning: slower, sometimes better on multi-step requests |
| `--temperature` | 0.3 | lower = more predictable tool use |
| `--max-steps` | 8 | max LLM calls per request |
| `--max-tool-chars` | 6000 | truncate tool results to protect the context window |
| `--llm-url`, `--mcp-url` | localhost:8080, localhost:8100/mcp | also via `LLM_URL`, `MCP_URL` |

**How the loop works:** the agent fetches the tool list from the MCP server, converts it to the
OpenAI tool format and sends the conversation to llama-server. When the model requests a tool call,
the agent runs it through MCP and feeds the result back, until the model answers in plain text.
Malformed arguments and tool errors are returned to the model as text so it can correct itself;
old turns are dropped when the history approaches the context limit.

## Classifier API

The classifier can also be used directly, without the agent.

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | – | status, model, execution providers, classes |
| `POST /predict?top=3` | form field `file` (one image) | top label + top-k probabilities |
| `POST /predict/batch?top=3` | form field `files`, repeated (max 32) | results in the order sent |

```bash
curl -s -F "file=@bottle.jpg" "localhost:8000/predict?top=3"
curl -s -F "files=@a.jpg" -F "files=@b.jpg" localhost:8000/predict/batch
```

```python
import requests
with open("bottle.jpg", "rb") as f:
    print(requests.post("http://localhost:8000/predict", files={"file": f}).json()["label"])
```

Interactive docs: http://localhost:8000/docs. Errors: `400` unreadable image or too many files,
`413` file over 10 MB, `422` wrong form field name.

## MCP tools

| Tool | Purpose |
|---|---|
| `list_classes()` | classes the classifier knows, and whether it runs on the GPU |
| `list_folder(path)` | subfolders with image counts, and the images in a folder |
| `classify_images(paths)` | up to 50 specific files: label, confidence, runner-up |
| `classify_folder(path, recursive, details)` | counts per class, mean confidence, low-confidence files |

- Paths are relative to `IMAGE_DIR` (full host paths under it also work). Anything outside is refused,
  and the folder is mounted read-only.
- Images are sent to the classifier in batches; a corrupt file is reported without failing the rest.
- The tools are few and coarse on purpose: small models are much more reliable when a request
  needs one tool call rather than ten.
- Any MCP client can use the server at `http://localhost:8100/mcp`. Requests with an unexpected
  `Host` header are rejected (DNS-rebinding protection).

## GPU memory budget

LLM and classifier share the 8 GB card, so every allocation is set explicitly:

| Item | Approx. VRAM | Setting |
|---|---|---|
| Qwen3 8B Q4_K_M weights | 4.7 GB | `LLM_MODEL` |
| KV cache, 12K tokens, 8-bit | 0.9 GB | `LLM_CTX`, `LLM_KV_TYPE` |
| llama.cpp buffers + CUDA context | ~0.6 GB | |
| Classifier (capped arena + CUDA context) | ~1.1 GB | `CLASSIFIER_GPU_MB` |
| Desktop / display | 0.1–0.4 GB | |
| **Total** | **~7.4–7.7 GB** | |

Check with `nvidia-smi`. If llama-server runs out of memory, lower `LLM_CTX` to 8192.

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `IMAGE_DIR` | – (required) | host folder the agent may read, mounted read-only |
| `UID`, `GID` | 1000 | your user/group id, so the MCP server can read your files |
| `LLM_MODEL` | `Qwen/Qwen3-8B-GGUF:Q4_K_M` | Hugging Face GGUF repo and quantization |
| `LLM_CTX` | 12288 | context window in tokens |
| `LLM_KV_TYPE` | `q8_0` | KV-cache precision (`f16` = more accurate, twice the memory) |
| `LLAMA_IMAGE_TAG` | `server-cuda13` | llama.cpp image; pin a release tag for reproducible builds |
| `CLASSIFIER_GPU_MB` | 768 | hard cap for the classifier's GPU memory |
| `CHUNK_SIZE` | 16 | images per request from the MCP server to the classifier |

Apply changes with `docker compose up -d`. To try a smaller, faster model:
`LLM_MODEL=Qwen/Qwen3-4B-GGUF:Q4_K_M` (~2.5 GB, weaker on multi-step requests).

## Troubleshooting

| Symptom | Fix |
|---|---|
| classifier restarting, `FileNotFoundError ... metadata.json` | no exported model in `app/model`: run `export_onnx.py` |
| `llm` shows "unhealthy" for minutes | normal on first start while the model downloads: `docker compose logs -f llm` |
| llama-server out of memory | lower `LLM_CTX` |
| classifier "Failed to allocate memory" | raise `CLASSIFIER_GPU_MB` or lower `CHUNK_SIZE` |
| `/health` lists only `CPUExecutionProvider` | GPU not reachable from the container: test `docker run --rm --gpus all ... nvidia-smi` |
| "outside the image folder" | the path isn't under `IMAGE_DIR` |
| permission denied reading images | `UID`/`GID` in `.env` must match `id -u` / `id -g` |
| model answers without calling a tool | try `--think` or a lower `--temperature` |
| "context size exceeded" | `/reset` in the agent, or raise `LLM_CTX` |

## Limitations

- RealWaste photos come from a fixed conveyor rig with consistent backgrounds. Expect lower accuracy
  on everyday phone photos; test on some of your own images, and add them to training if the gap is large.
- An 8B model handles one- or two-step requests reliably. Long multi-step plans are where it
  starts to fail; that's a limit of the model size that fits in 8 GB.

## Credits and license

- Dataset: **RealWaste** by S. Single, S. Iranmanesh and R. Raad, licensed CC BY 4.0.
  Paper: [*RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning*](https://www.mdpi.com/2078-2489/14/12/633),
  Information 14(12), 633, 2023. The dataset is not redistributed here.
- Pretrained backbone: [timm](https://github.com/huggingface/pytorch-image-models) ConvNeXt-Tiny.
- LLM: [Qwen3](https://huggingface.co/Qwen) (Apache 2.0), served with [llama.cpp](https://github.com/ggml-org/llama.cpp).
- [Model Context Protocol](https://modelcontextprotocol.io) Python SDK.
