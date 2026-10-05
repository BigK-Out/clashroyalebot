"""Train PlayNet on dataset/selfplay_clips, validate on held-out matches, export ONNX.

  .venv/bin/python train_plays.py [--epochs 12]
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from selfplay.clips import CLASSES
from selfplay.model import PlayNet

CLIPS = Path("../dataset/selfplay_clips")
OUT = Path("runs/plays")


def is_val(match_id):
    return int(hashlib.md5(match_id.encode()).hexdigest(), 16) % 10 == 0


class Clips(Dataset):
    def __init__(self, files, augment):
        self.items, self.augment = [], augment
        self.data = {}
        for f in files:
            d = np.load(f)
            # Unpack the clips once to a plain .npy next to the npz and memory-map it:
            # the whole dataset does not fit in RAM. Then drop the pixels from the npz (they
            # would be stored twice); it keeps only the labels.
            raw = f.with_suffix(".X.npy")
            if "X" in d.files:
                if not raw.exists():
                    np.save(raw, d["X"])
                labels = {k: d[k] for k in d.files if k != "X"}
                np.savez(f, **labels)
                d = np.load(f)
            self.data[f] = (np.load(raw, mmap_mode="r"), d["y"], d["t"])
            self.items += [(f, i) for i in range(len(d["y"]))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        f, i = self.items[k]
        X, y, t = self.data[f]
        x = torch.from_numpy(np.array(X[i])).permute(0, 3, 1, 2).float() / 255  # [8,3,128,128]
        if self.augment:
            x = x * (0.8 + 0.4 * torch.rand(1)) + 0.1 * (torch.rand(1) - 0.5)
            dx, dy = np.random.randint(-8, 9, 2)
            x = torch.roll(x, (int(dy), int(dx)), (2, 3))
            x = x.clamp(0, 1)
        return x, int(y[i]), f.stem, float(t[i])


def save_checkpoint(path, net, opt, sched, epoch, best, files=None):
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "best": best, "files": files}, path)


def load_checkpoint(path, net, opt, sched, files=None):
    """Restores everything; returns (next epoch, best val accuracy so far). A checkpoint
    trained on other clip files (an earlier batch) is not resumed: returns (0, 0.0)."""
    c = torch.load(path, map_location="cpu")
    if files is not None and c.get("files") != files:
        return 0, 0.0
    net.load_state_dict(c["net"])
    opt.load_state_dict(c["opt"])
    sched.load_state_dict(c["sched"])
    return c["epoch"] + 1, c["best"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--resume", action="store_true", help="continue from runs/plays/last.pt")
    ap.add_argument("--eval-only", action="store_true", help="score runs/plays/best.pt on the validation set")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    files = sorted(CLIPS.glob("*_note*.npz"))
    val = Clips([f for f in files if is_val(f.name.rsplit("_", 1)[0])], False)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.eval_only:
        net = PlayNet(len(CLASSES), pretrained=False).to(dev)
        net.load_state_dict(torch.load(OUT / "best.pt", map_location=dev))
        acc2, acc05, preds = evaluate(net, val, dev)
        np.savez(OUT / "val_preds.npz", **preds)
        print(f"best.pt: val top-1 @2s {acc2:.4f} @0.5s {acc05:.4f}", flush=True)
        return
    train = Clips([f for f in files if not is_val(f.name.rsplit("_", 1)[0])], True)
    print(f"train {len(train)} clips, val {len(val)} clips, {len(files)} files", flush=True)
    counts = np.bincount([train.data[f][1][i] for f, i in train.items], minlength=len(CLASSES))
    weights = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32).to(dev)
    net = PlayNet(len(CLASSES)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.epochs * (len(train) // a.batch + 1))
    # Mixed precision and a small batch keep the laptop GPU's load down (a full-load
    # run crashed the driver); a checkpoint every epoch makes a crash cost one epoch.
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    dl = DataLoader(train, a.batch, shuffle=True, num_workers=2, drop_last=True)
    names = sorted({f.name for f, _ in train.items})
    start, best = (load_checkpoint(OUT / "last.pt", net, opt, sched, names) if a.resume and (OUT / "last.pt").exists() else (0, 0.0))
    for ep in range(start, a.epochs):
        net.train()
        for x, y, _, _ in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                # Full clip and the early (first 4 frames) readout share the weights.
                loss = F.cross_entropy(net(x), y, weight=weights) + 0.5 * F.cross_entropy(net(x[:, :4]), y, weight=weights)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        acc2, acc05, preds = evaluate(net, val, dev)
        print(f"epoch {ep}: val top-1 @2s {acc2:.4f} @0.5s {acc05:.4f}", flush=True)
        if acc2 >= best:
            best = acc2
            torch.save(net.state_dict(), OUT / "best.pt")
            np.savez(OUT / "val_preds.npz", **preds)
        save_checkpoint(OUT / "last.pt", net, opt, sched, ep, best, names)
    net.load_state_dict(torch.load(OUT / "best.pt", map_location="cpu"))
    export(net.cpu().eval())


@torch.no_grad()
def evaluate(net, val, dev):
    net.eval()
    p2, p05, ys, ms, ts = [], [], [], [], []
    for x, y, m, t in DataLoader(val, 32, num_workers=2):
        x = x.to(dev)
        p2.append(torch.softmax(net(x), 1).cpu())
        p05.append(torch.softmax(net(x[:, :4]), 1).cpu())
        ys.append(y)
        ms += list(m)
        ts.append(t)
    p2, p05, ys = torch.cat(p2).numpy(), torch.cat(p05).numpy(), torch.cat(ys).numpy()
    plays = ys != len(CLASSES) - 1
    acc = lambda p: float((p.argmax(1)[plays] == ys[plays]).mean()) if plays.any() else 0.0
    return acc(p2), acc(p05), {"p2s": p2, "p05": p05, "y": ys, "match": np.array(ms), "t": torch.cat(ts).numpy()}


def export(net):
    Path("../assets/models").mkdir(exist_ok=True)
    torch.onnx.export(net, torch.rand(1, 8, 3, 128, 128), "../assets/models/plays.onnx", input_names=["clip"],
                      output_names=["logits"], dynamic_axes={"clip": {1: "frames"}}, opset_version=17)
    Path("../assets/models/plays.txt").write_text("\n".join(CLASSES) + "\n")
    print("exported ../assets/models/plays.onnx")


if __name__ == "__main__":
    main()
