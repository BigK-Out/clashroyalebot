"""Spec section 4 metrics on held-out matches -> runs/plays/report.md."""
import collections
from pathlib import Path

import numpy as np

from selfplay.clips import CLASSES
from selfplay.deck import refine

NP = len(CLASSES) - 1
d = np.load("runs/plays/val_preds.npz")
y, match, t = d["y"], d["match"], d["t"]
lines = ["# Play classifier report", ""]
for name, p in (("0.5 s", d["p05"]), ("2 s", d["p2s"])):
    plays = y != NP
    raw = p.argmax(1)
    deck = raw.copy()
    for m in np.unique(match):
        idx = np.where((match == m) & plays)[0]
        idx = idx[np.argsort(t[idx])]
        if len(idx):
            deck[idx] = refine(p[idx], NP)
    lines.append(f"- top-1 @ {name}: {np.mean(raw[plays] == y[plays]):.4f} raw, {np.mean(deck[plays] == y[plays]):.4f} with deck inference ({plays.sum()} plays)")
p2 = d["p2s"].argmax(1)
tp = np.sum((p2 == NP) & (y == NP))
lines.append(f"- no_play precision {tp / max(1, np.sum(p2 == NP)):.3f}, recall {tp / max(1, np.sum(y == NP)):.3f}")
conf = collections.Counter((CLASSES[a], CLASSES[b]) for a, b in zip(y, p2) if a != b)
lines += ["", "## Top confusions (true -> predicted)", ""] + [f"- {a} -> {b}: {n}" for (a, b), n in conf.most_common(15)]
per = {CLASSES[k]: np.mean(p2[y == k] == k) for k in np.unique(y) if k != NP}
lines += ["", "## Worst cards", ""] + [f"- {c}: {a:.3f}" for c, a in sorted(per.items(), key=lambda x: x[1])[:20]]
rec = Path("runs/recall.log")
if rec.exists():
    lines += ["", "## Proposal recall", ""] + [l for l in rec.read_text().splitlines() if l.startswith(("recall", "overall"))]
text = "\n".join(lines) + "\n"
Path("runs/plays/report.md").write_text(text)
print(text)
