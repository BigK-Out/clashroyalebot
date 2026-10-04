"""Recall of logged enemy plays by proposals, per card kind, on aligned matches."""
import collections
import json
from pathlib import Path

from selfplay.proposals import enemy_plays, match_props, video_proposals

kind = {c["slug"]: c["type"] for c in json.load(open("../assets/cards.json"))}
hit, total, phantom, nprops = collections.Counter(), collections.Counter(), 0, 0
for m in sorted(Path("../dataset/selfplay").iterdir()):
    a = m / "align.json"
    if not a.exists():
        continue
    align = json.load(open(a))
    for viewer in ("note9", "note14"):
        if not align.get(viewer):
            continue
        props = video_proposals(m, viewer)
        plays = enemy_plays(m, viewer)
        pairs, unmatched = match_props(props, plays)
        for i, p in enumerate(plays):
            total[kind[p["card"]]] += 1
            hit[kind[p["card"]]] += i in pairs
        phantom += len(unmatched)
        nprops += len(props)
    print(m.name, dict(total), dict(hit), flush=True)
print("recall by kind:", {k: f"{hit[k]}/{total[k]} = {hit[k] / total[k]:.3f}" for k in total})
print("overall recall:", sum(hit.values()) / max(1, sum(total.values())), "| proposals:", nprops, "phantom:", phantom)
