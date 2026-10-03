"""Group unit crops by appearance so whole groups can be named at once.

dataset/units/*.png -> ResNet-18 (ImageNet) features -> k-means -> dataset/units_clusters/
  clusters.csv (file,cluster), one montage per cluster (cluster_XX.png) for naming.
"""
import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--units", default="../dataset/units")
ap.add_argument("--out", default="../dataset/units_clusters")
ap.add_argument("--k", type=int, default=40)
args = ap.parse_args()

files = sorted(Path(args.units).glob("*.png"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
model = torchvision.models.resnet18(weights=w)
model.fc = torch.nn.Identity()
model.eval().to(dev)
tf = w.transforms()

feats = []
with torch.no_grad():
    for i in range(0, len(files), 256):
        batch = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + 256]]).to(dev)
        feats.append(torch.nn.functional.normalize(model(batch), dim=1).cpu())
x = torch.cat(feats)

# k-means (cosine on normalized features), k-means++ init, fixed seed.
g = torch.Generator().manual_seed(0)
cent = x[torch.randint(len(x), (1,), generator=g)]
for _ in range(args.k - 1):
    d = (1 - x @ cent.T).min(dim=1).values.clamp(min=0)
    cent = torch.cat([cent, x[torch.multinomial(d, 1, generator=g)]])
for _ in range(50):
    assign = (x @ cent.T).argmax(dim=1)
    cent = torch.stack([torch.nn.functional.normalize(x[assign == c].mean(0), dim=0) if (assign == c).any() else cent[c]
                        for c in range(args.k)])

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "clusters.csv").write_text("file,cluster\n" + "".join(f"{f.name},{int(a)}\n" for f, a in zip(files, assign)))
sizes = torch.bincount(assign, minlength=args.k)
for c in range(args.k):
    members = [f for f, a in zip(files, assign) if a == c]
    # Most typical first (closest to the centroid).
    members.sort(key=lambda f: -float(x[files.index(f)] @ cent[c]))
    pick = members[:24]
    m = Image.new("RGB", (8 * 66, 3 * 66 + 18), (30, 30, 30))
    for i, f in enumerate(pick):
        m.paste(Image.open(f).convert("RGB"), ((i % 8) * 66, 18 + (i // 8) * 66))
    ImageDraw.Draw(m).text((4, 3), f"cluster {c}  n={len(members)}", fill=(255, 255, 0))
    m.save(out / f"cluster_{c:02d}.png")
print("cluster sizes:", sizes.tolist())
