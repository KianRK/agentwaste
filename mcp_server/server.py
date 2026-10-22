"""MCP server that exposes the RealWaste classifier as agent tools.

Transport: streamable HTTP at http://<host>:8100/mcp (stateless, JSON responses). MCP Python SDK v2.
The server never loads the model; it reads images from a read-only mounted folder
and forwards them to the classifier API in batches.

Environment:
    CLASSIFIER_URL   classifier API base URL          (default http://classifier:8000)
    IMAGE_ROOT       image folder inside the container (default /images)
    HOST_IMAGE_ROOT  same folder's path on the host, so the agent can use host paths too
    CHUNK_SIZE       images per request to the classifier (default 16, must not exceed the classifier's MAX_BATCH)
    MAX_FILES        refuse folders with more images than this (default 2000)
    LOW_CONFIDENCE   top-1 probability below which a result is flagged (default 0.6)
    PORT             listen port (default 8100)
"""
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated

import httpx2
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000").rstrip("/")
IMAGE_ROOT = Path(os.getenv("IMAGE_ROOT", "/images")).resolve()
HOST_IMAGE_ROOT = os.getenv("HOST_IMAGE_ROOT", "").rstrip("/")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "16"))
MAX_FILES = int(os.getenv("MAX_FILES", "2000"))
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.6"))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_shown_root = HOST_IMAGE_ROOT or str(IMAGE_ROOT)
mcp = MCPServer(
    "realwaste",
    instructions=(
        "Tools for classifying photos of waste into material classes with a dedicated image classifier. "
        f"All images live in the image folder {_shown_root}. Pass paths relative to that folder "
        f"(e.g. 'bin_3' or 'bin_3/img_01.jpg'); full paths starting with {_shown_root} also work. "
        "Use list_folder to explore, classify_folder for whole folders, classify_images for specific files. "
        "Never guess a class without calling a tool."
    ),
)


# --------------------------------------------------------------------------- helpers
def resolve(path: str) -> Path:
    """Map an agent-supplied path to a location inside IMAGE_ROOT, refusing anything outside it."""
    p = (path or ".").strip()
    for prefix in filter(None, (HOST_IMAGE_ROOT, str(IMAGE_ROOT))):
        if p == prefix or p.startswith(prefix + "/"):
            p = p[len(prefix):]
            break
    else:
        if p.startswith(("/", "~")):
            raise ToolError(f"'{path}' is outside the image folder {_shown_root}. "
                             "Use a path relative to that folder.")
    full = (IMAGE_ROOT / p.lstrip("/")).resolve()
    if not full.is_relative_to(IMAGE_ROOT):
        raise ToolError(f"'{path}' is outside the image folder {_shown_root}.")
    if not full.exists():
        raise ToolError(f"'{path}' does not exist in the image folder.")
    return full


def rel(p: Path) -> str:
    return str(p.relative_to(IMAGE_ROOT)) or "."


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def images_in(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if is_image(p))


def _result(path: Path, preds: list[dict]) -> dict:
    out = {"path": rel(path), "label": preds[0]["label"], "confidence": preds[0]["probability"]}
    if len(preds) > 1:
        out["runner_up"] = preds[1]["label"]
        out["runner_up_confidence"] = preds[1]["probability"]
    return out


def _detail(r: httpx2.Response) -> str:
    try:
        return str(r.json().get("detail", r.text))[:200]
    except ValueError:
        return r.text[:200]


async def classify_files(files: list[Path]) -> tuple[list[dict], list[dict]]:
    """Send files to the classifier in chunks. A chunk rejected with 4xx (e.g. one corrupt file)
    is retried file by file so one bad image doesn't fail the rest."""
    results, errors = [], []
    async with httpx2.AsyncClient(base_url=CLASSIFIER_URL, timeout=120) as client:
        async def post(chunk: list[Path]) -> httpx2.Response:
            parts = [("files", (f.name, f.read_bytes(), "application/octet-stream")) for f in chunk]
            return await client.post("/predict/batch", files=parts, params={"top": 2})

        for i in range(0, len(files), CHUNK_SIZE):
            chunk = files[i:i + CHUNK_SIZE]
            try:
                r = await post(chunk)
            except httpx2.TransportError as e:
                raise ToolError(f"classifier not reachable at {CLASSIFIER_URL}: {e}") from e
            if r.status_code == 200:
                results += [_result(f, res["predictions"]) for f, res in zip(chunk, r.json()["results"])]
                continue
            if r.status_code >= 500 or len(chunk) == 1:
                if r.status_code >= 500:
                    raise ToolError(f"classifier error {r.status_code}: {r.text[:300]}")
                errors.append({"path": rel(chunk[0]), "error": _detail(r)})
                continue
            for f in chunk:
                r1 = await post([f])
                if r1.status_code == 200:
                    results.append(_result(f, r1.json()["results"][0]["predictions"]))
                else:
                    errors.append({"path": rel(f), "error": _detail(r1)})
    return results, errors


# --------------------------------------------------------------------------- tools
@mcp.tool()
async def list_classes() -> dict:
    """Return the material classes the classifier can recognise and whether it runs on the GPU."""
    try:
        async with httpx2.AsyncClient(base_url=CLASSIFIER_URL, timeout=10) as client:
            h = (await client.get("/health")).json()
    except httpx2.TransportError as e:
        raise ToolError(f"classifier not reachable at {CLASSIFIER_URL}: {e}") from e
    return {"classes": h["classes"], "model": h["model"],
            "on_gpu": "CUDAExecutionProvider" in h.get("providers", [])}


@mcp.tool()
def list_folder(
    path: Annotated[str, Field(description="Folder relative to the image folder; '.' for the top level")] = ".",
) -> dict:
    """List the subfolders of a folder (with their image counts) and the images directly inside it."""
    folder = resolve(path)
    if not folder.is_dir():
        raise ToolError(f"'{path}' is a file, not a folder.")
    subs = sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
    imgs = images_in(folder, recursive=False)
    return {
        "folder": rel(folder),
        "subfolders": [{"name": s.name, "images": len(images_in(s, recursive=True))} for s in subs],
        "images": len(imgs),
        "sample_files": [p.name for p in imgs[:10]],
    }


@mcp.tool()
async def classify_images(
    paths: Annotated[list[str], Field(description="Image file paths relative to the image folder (max 50)")],
) -> dict:
    """Classify specific image files. Returns label, confidence and runner-up for each file."""
    if len(paths) > 50:
        raise ToolError("At most 50 files per call; use classify_folder for more.")
    files = []
    for p in paths:
        f = resolve(p)
        if not is_image(f):
            raise ToolError(f"'{p}' is not an image file.")
        files.append(f)
    results, errors = await classify_files(files)
    for r in results:
        r["low_confidence"] = r["confidence"] < LOW_CONFIDENCE
    return {"results": results, "errors": errors}


@mcp.tool()
async def classify_folder(
    path: Annotated[str, Field(description="Folder relative to the image folder; '.' for the top level")] = ".",
    recursive: Annotated[bool, Field(description="Include images in subfolders")] = False,
    details: Annotated[bool, Field(description="Also return the label of every file (long output)")] = False,
) -> dict:
    """Classify every image in a folder. Returns counts per class, average confidence per class
    and the files with low confidence. Set details=true only if the user needs per-file labels."""
    folder = resolve(path)
    if not folder.is_dir():
        raise ToolError(f"'{path}' is a file; use classify_images for single files.")
    files = images_in(folder, recursive)
    if not files:
        return {"folder": rel(folder), "images": 0, "note": "no images found"
                + ("" if recursive else "; try recursive=true")}
    if len(files) > MAX_FILES:
        raise ToolError(f"{len(files)} images exceed the limit of {MAX_FILES}; classify a subfolder instead.")

    results, errors = await classify_files(files)
    counts = Counter(r["label"] for r in results)
    conf = defaultdict(list)
    for r in results:
        conf[r["label"]].append(r["confidence"])
    low = sorted((r for r in results if r["confidence"] < LOW_CONFIDENCE), key=lambda r: r["confidence"])

    out = {
        "folder": rel(folder),
        "images": len(files),
        "classified": len(results),
        "counts": dict(counts.most_common()),
        "mean_confidence": {k: round(sum(v) / len(v), 3) for k, v in conf.items()},
        "low_confidence_threshold": LOW_CONFIDENCE,
        "low_confidence_total": len(low),
        "low_confidence": low[:15],
        "errors": errors,
    }
    if details:
        out["files"] = [{"path": r["path"], "label": r["label"], "confidence": r["confidence"]}
                        for r in results[:300]]
        if len(results) > 300:
            out["files_truncated"] = True
    return out


if __name__ == "__main__":
    mcp.run(
        "streamable-http",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8100")),
        stateless_http=True,   # no per-client session state: each tool call is independent
        json_response=True,    # plain JSON responses instead of SSE streams
        # Reject requests whose Host header isn't expected (DNS-rebinding protection).
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[h.strip() for h in os.getenv(
                "ALLOWED_HOSTS", "localhost:*,127.0.0.1:*,mcp:*").split(",")],
            allowed_origins=[],
        ),
    )
