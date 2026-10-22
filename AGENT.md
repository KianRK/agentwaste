# Local agent with MCP access to the classifier

```
 you ─► agent/agent.py (host, Python) ──OpenAI API──► llm         llama.cpp llama-server, Qwen3 8B   :8080
                    │
                    └──MCP (streamable HTTP)────────► mcp         realwaste MCP server               :8100
                                                       │ HTTP
                                                       └────────► classifier  FastAPI + ONNX Runtime :8000
```

All three services run from `compose.yaml` and share the RTX 4060. Ports are bound to `127.0.0.1` only.
The agent loop is plain Python with no agent framework. It calls llama-server's `/v1/chat/completions`
directly over HTTP (`httpx2`), so every request and response field is visible, including llama.cpp's
`timings`. It talks to the MCP server with the official MCP Python SDK v2 (`mcp.Client`). The loop
between them is about 100 lines you control.

## GPU memory budget (8 GB)

| Item | Approx. VRAM | Set by |
|---|---|---|
| Qwen3 8B Q4_K_M weights | 4.7 GB | `LLM_MODEL` |
| KV cache, 12K tokens, q8_0 | 0.9 GB | `LLM_CTX`, `LLM_KV_TYPE` |
| llama.cpp compute buffers + CUDA context | ~0.6 GB | |
| Classifier arena cap + CUDA context | ~1.1 GB | `CLASSIFIER_GPU_MB` |
| Desktop / display | 0.1–0.4 GB | |
| **Total** | **~7.4–7.7 GB** | |

Check the real numbers with `nvidia-smi` after `docker compose up`. If llama-server fails at startup
with an out-of-memory error, set `LLM_CTX=8192`. If you have headroom, raise `LLM_CTX` to 16384.

## First start

```bash
cd ~/Projects/realwaste
docker rm -f realwaste                 # the old standalone classifier container (port 8000)
cp .env.example .env                    # then edit IMAGE_DIR (and UID/GID if not 1000)
mkdir -p models
docker compose up -d --build
docker compose logs -f llm              # first start downloads ~5 GB; wait for "server is listening"
```

Agent environment (once):

```bash
python3 -m venv agent/.venv
agent/.venv/bin/pip install -r agent/requirements.txt
```

## Use the agent

```bash
agent/.venv/bin/python agent/agent.py                       # interactive
agent/.venv/bin/python agent/agent.py -v                    # also show tool results, token counts, tok/s
agent/.venv/bin/python agent/agent.py "summarise folder bin_3"   # one task, prints the answer, exits
```

Interactive commands: `/tools`, `/reset`, `/exit`.

Example requests:
- "What folders do I have?"
- "Classify everything in bin_3 and tell me which images you're unsure about."
- "Is kitchen/photo_12.jpg recyclable?"

Agent options: `--think` (Qwen3 reasoning on: slower, sometimes better for multi-step requests),
`--max-steps`, `--temperature`, `--max-tool-chars`, `--llm-url`, `--mcp-url`.

## MCP tools

| Tool | Purpose |
|---|---|
| `list_classes` | Classes the classifier knows, and whether it runs on the GPU |
| `list_folder(path)` | Subfolders with image counts, images in a folder |
| `classify_images(paths)` | Up to 50 specific files: label, confidence, runner-up |
| `classify_folder(path, recursive, details)` | Counts per class, mean confidence, low-confidence files |

Paths are relative to `IMAGE_DIR`; full host paths under `IMAGE_DIR` work too. Anything outside is refused.
Expected failures (bad path, classifier down) come back to the model as readable tool errors, so it can correct itself.
The MCP server can be used by any other MCP client at `http://localhost:8100/mcp`. It only accepts requests
whose Host header is `localhost`, `127.0.0.1` or `mcp` (DNS-rebinding protection; set `ALLOWED_HOSTS` to change).

## Operating

```bash
docker compose ps                       # status
docker compose logs -f mcp              # tool calls as they arrive
docker compose restart classifier       # after exporting a new model
docker compose up -d                    # after changing .env (recreates affected services)
docker compose up -d --build mcp        # after editing mcp_server/server.py
docker compose down                     # stop everything
nvidia-smi                              # VRAM per process
curl -s localhost:8080/metrics | grep -E "tokens_predicted|prompt_tokens"   # LLM throughput
```

llama-server also has a built-in chat page at http://localhost:8080 for testing the model without tools.

## Swapping the model

Set `LLM_MODEL` to another GGUF repo and quant, then `docker compose up -d llm`. Examples:

| `LLM_MODEL` | VRAM | Notes |
|---|---|---|
| `Qwen/Qwen3-8B-GGUF:Q4_K_M` | ~4.7 GB | default: best tool calling that fits |
| `Qwen/Qwen3-4B-GGUF:Q4_K_M` | ~2.5 GB | about twice as fast; weaker on multi-step requests; allows `LLM_CTX=32768` |

## Troubleshooting

- **Agent waits forever for llama-server:** `docker compose logs llm`. Out of memory: lower `LLM_CTX`.
- **Classifier errors with "Failed to allocate memory":** raise `CLASSIFIER_GPU_MB` or lower `CHUNK_SIZE`.
- **"outside the image folder":** the path isn't under `IMAGE_DIR`. Change `IMAGE_DIR` and `docker compose up -d`.
- **Permission denied reading images:** `UID`/`GID` in `.env` must match `id -u` / `id -g`.
- **Model answers without calling a tool:** try `--think`, or a lower `--temperature`.
- **"context size exceeded" from llama-server:** `/reset` in the agent, or raise `LLM_CTX` if VRAM allows.

## Versions

MCP Python SDK v2 (`mcp>=2.2,<3`, which uses `httpx2`), llama.cpp `server-cuda13` image,
`onnxruntime-gpu==1.31.0` on CUDA 13. For reproducible builds, pin the llama.cpp image to a
release tag (e.g. `LLAMA_IMAGE_TAG=server-cuda13-b<build>`) once you have a version that works.
