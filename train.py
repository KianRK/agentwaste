#!/usr/bin/env python3
"""Fine-tune a pretrained timm backbone on the RealWaste dataset.

Phase 1: train only the classifier head (backbone frozen).
Phase 2: unfreeze everything, low LR, warmup + cosine decay.
Best checkpoint is chosen by validation macro-F1, then evaluated once on the test split.

Usage:
    python train.py --data-dir data --out-dir runs/convnext_tiny
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- data
def find_class_root(root: Path) -> Path:
    """Return the first directory (shallowest first) whose subfolders contain images.
    Lets --data-dir point at the unzipped Kaggle folder without knowing its exact nesting."""
    dirs = [root] + sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts))
    for d in dirs:
        subs = [s for s in d.iterdir() if s.is_dir()]
        with_imgs = [s for s in subs if any(f.suffix.lower() in IMG_EXTS for f in s.iterdir())]
        if len(with_imgs) >= 2:
            return d
    raise SystemExit(f"No folder with class subfolders of images found under {root}")


def list_images(class_root: Path):
    classes = sorted(
        d.name for d in class_root.iterdir()
        if d.is_dir() and any(f.suffix.lower() in IMG_EXTS for f in d.iterdir())
    )
    items = [
        (str(f), idx)
        for idx, c in enumerate(classes)
        for f in sorted((class_root / c).iterdir())
        if f.suffix.lower() in IMG_EXTS
    ]
    return classes, items


class ImageList(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
        return self.tf(im), label


def build_transforms(size, mean, std):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),          # waste has no canonical orientation
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])
    # NOTE: the API (app/main.py) reproduces exactly this eval transform with PIL + numpy.
    eval_tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# --------------------------------------------------------------------------- training
def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    use_amp = device.type == "cuda"
    loss_sum, n = 0.0, 0
    preds, targets = [], []
    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": loss_sum / n,
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }, preds, targets


def warmup_cosine(optimizer, warmup_steps, total_steps, min_ratio=0.01):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/convnext_tiny"))
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k",
                    help="any timm model name, e.g. efficientnetv2_rw_s.ra2_in1k, mobilenetv3_large_100.ra_in1k")
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=25, help="full fine-tuning epochs after the head phase")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, training on CPU (slow).")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data + stratified 70/15/15 split
    class_root = find_class_root(args.data_dir)
    classes, items = list_images(class_root)
    labels = [y for _, y in items]
    print(f"Found {len(items)} images in {len(classes)} classes under {class_root}")
    for c, k in sorted(Counter(labels).items()):
        print(f"  {classes[c]:<22} {k}")

    train_items, rest = train_test_split(items, test_size=0.30, stratify=labels, random_state=args.seed)
    val_items, test_items = train_test_split(rest, test_size=0.50, stratify=[y for _, y in rest],
                                             random_state=args.seed)
    (args.out_dir / "splits.json").write_text(json.dumps(
        {"train": train_items, "val": val_items, "test": test_items}, indent=0))

    # ---- model
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=len(classes),
                              drop_rate=0.2, drop_path_rate=0.1)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean, std = list(cfg["mean"]), list(cfg["std"])
    model = model.to(device).to(memory_format=torch.channels_last)

    train_tf, eval_tf = build_transforms(args.img_size, mean, std)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_dl = DataLoader(ImageList(train_items, train_tf), shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(ImageList(val_items, eval_tf), shuffle=False, **loader_kw)
    test_dl = DataLoader(ImageList(test_items, eval_tf), shuffle=False, **loader_kw)

    # ---- class-weighted loss for the imbalance (plastic/metal vs textile)
    counts = np.bincount([y for _, y in train_items], minlength=len(classes))
    weights = torch.tensor(len(train_items) / (len(classes) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    meta = {"model": args.model, "classes": classes, "img_size": args.img_size, "mean": mean, "std": std}
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    best_f1, history = -1.0, []

    def log_and_maybe_save(phase, epoch, tr, va, t0):
        nonlocal best_f1
        history.append({"phase": phase, "epoch": epoch, "train": tr, "val": va})
        flag = ""
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            flag = "  *saved*"
        print(f"[{phase} {epoch:>2}] train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.3f} f1 {va['macro_f1']:.3f} "
              f"({time.time() - t0:.0f}s){flag}")

    # ---- phase 1: head only
    for p in model.parameters():
        p.requires_grad = False
    head_params = list(model.get_classifier().parameters())
    for p in head_params:
        p.requires_grad = True
    opt = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
    for ep in range(1, args.head_epochs + 1):
        t0 = time.time()
        tr, _, _ = run_epoch(model, train_dl, criterion, device, opt)
        va, _, _ = run_epoch(model, val_dl, criterion, device)
        log_and_maybe_save("head", ep, tr, va, t0)

    # ---- phase 2: full fine-tune
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train_dl)
    sched = warmup_cosine(opt, warmup_steps=len(train_dl), total_steps=steps)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr, _, _ = run_epoch(model, train_dl, criterion, device, opt, sched)
        va, _, _ = run_epoch(model, val_dl, criterion, device)
        log_and_maybe_save("full", ep, tr, va, t0)

    # ---- final test evaluation with the best checkpoint
    model.load_state_dict(torch.load(args.out_dir / "best.pt", map_location=device))
    te, preds, targets = run_epoch(model, test_dl, criterion, device)
    report = classification_report(targets, preds, labels=list(range(len(classes))),
                                   target_names=classes, digits=3, zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    print("\nTEST:", json.dumps(te))
    print(report)
    print("Confusion matrix (rows = true, cols = predicted):")
    print("\n".join(f"{classes[i][:14]:<15}" + " ".join(f"{v:>4}" for v in row) for i, row in enumerate(cm)))
    (args.out_dir / "metrics.json").write_text(json.dumps(
        {"best_val_macro_f1": best_f1, "test": te, "confusion_matrix": cm.tolist(), "history": history},
        indent=2))
    print(f"\nBest checkpoint: {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
