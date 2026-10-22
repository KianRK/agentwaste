#!/usr/bin/env python3
"""Export the best checkpoint to ONNX (dynamic batch) and verify it against PyTorch.

Usage:
    python export_onnx.py --run-dir runs/convnext_tiny --out-dir app/model
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/convnext_tiny"))
    ap.add_argument("--out-dir", type=Path, default=Path("app/model"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    meta = json.loads((args.run_dir / "metadata.json").read_text())
    model = timm.create_model(meta["model"], pretrained=False, num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(args.run_dir / "best.pt", map_location="cpu"))
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "model.onnx"
    size = meta["img_size"]
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, dynamo=False,
    )
    shutil.copy(args.run_dir / "metadata.json", args.out_dir / "metadata.json")

    # verify: batch of 4 also checks that the dynamic batch axis works
    x = torch.randn(4, 3, size, size)
    with torch.inference_mode():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    same_argmax = bool((ref.argmax(1) == out.argmax(1)).all())
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB); "
          f"max |torch - onnx| = {diff:.2e}, same predictions: {same_argmax}")
    if diff > 1e-3 or not same_argmax:
        raise SystemExit("ONNX output does not match PyTorch, check the export.")


if __name__ == "__main__":
    main()
