"""Train the unit-type classifier on named clusters and export ONNX.

dataset/units/*.png + dataset/units_clusters/clusters.csv + cluster_names.toml
-> runs/units/best.pt, ../assets/models/units.onnx (input 1x3x64x64 float 0..1, output logits),
   ../assets/models/units.txt (class names, one per line, index = logit).
Split by recording time (frame id): the newest 20% of frames validate, so near-duplicate
consecutive frames can't leak between train and val.
"""
import csv, tomllib, random
from pathlib import Path
import torch, torchvision
from PIL import Image

root = Path("../dataset")
names = {int(k): v for k, v in tomllib.load(open("cluster_names.toml", "rb"))["names"].items()}
rows = list(csv.DictReader(open(root / "units_clusters/clusters.csv")))
items = [(r["file"], names[int(r["cluster"])]) for r in rows if int(r["cluster"]) in names]
classes = sorted({c for _, c in items})
cidx = {c: i for i, c in enumerate(classes)}
frames = sorted({f.split("_")[0] for f, _ in items})
val_frames = set(frames[int(len(frames) * 0.8):])
train = [(f, cidx[c]) for f, c in items if f.split("_")[0] not in val_frames]
val = [(f, cidx[c]) for f, c in items if f.split("_")[0] in val_frames]
print(f"{len(classes)} classes, {len(train)} train / {len(val)} val")

tf_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(64, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(0.2, 0.2, 0.1, 0.0),  # no hue: color carries meaning
    torchvision.transforms.ToTensor(),
])
tf_val = torchvision.transforms.ToTensor()

class DS(torch.utils.data.Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        f, y = self.items[i]
        return self.tf(Image.open(root / "units" / f).convert("RGB")), y

# Balance classes: cannon and junk dominate.
counts = torch.bincount(torch.tensor([y for _, y in train]), minlength=len(classes)).float()
weights = [1.0 / counts[y].item() for _, y in train]
sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
dl = torch.utils.data.DataLoader(DS(train, tf_train), batch_size=128, sampler=sampler, num_workers=4)
vl = torch.utils.data.DataLoader(DS(val, tf_val), batch_size=256, num_workers=4)

dev = "cuda"
model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(classes))
model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 40
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(dl))
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def evaluate():
    model.eval(); correct = torch.zeros(len(classes)); total = torch.zeros(len(classes))
    with torch.no_grad():
        for x, y in vl:
            p = model((x.to(dev) - mean) / std).argmax(1).cpu()
            for c in range(len(classes)):
                m = y == c; total[c] += m.sum(); correct[c] += (p[m] == c).sum()
    return correct, total

import sys
best = 0.0
Path("runs/units").mkdir(parents=True, exist_ok=True)
for ep in range(0 if "--export-only" in sys.argv else epochs):
    model.train()
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        loss = torch.nn.functional.cross_entropy(model((x - mean) / std), y, label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    c, t = evaluate()
    acc = (c.sum() / t.sum()).item()
    if acc >= best:
        best = acc; torch.save(model.state_dict(), "runs/units/best.pt")
    if ep % 5 == 4 or ep == epochs - 1:
        print(f"epoch {ep+1}: val acc {acc:.3f} (best {best:.3f})")

model.load_state_dict(torch.load("runs/units/best.pt")); c, t = evaluate()
print("per class (val):")
for i, n in enumerate(classes):
    if t[i] > 0: print(f"  {n:16s} {int(c[i])}/{int(t[i])}  {c[i]/t[i]:.2f}")

# Export with normalization baked in: Rust feeds plain 0..1 RGB.
class Wrapped(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m((x - mean.cpu()) / std.cpu())
w = Wrapped(model.cpu().eval())
out = Path("../assets/models"); out.mkdir(parents=True, exist_ok=True)
torch.onnx.export(w, torch.zeros(1, 3, 64, 64), out / "units.onnx", input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17)
(out / "units.txt").write_text("\n".join(classes) + "\n")
print("exported", out / "units.onnx")
