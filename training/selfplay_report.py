"""Summarize the self-play dataset: coverage per card, label quality, timing."""
import collections
import json
import statistics
from pathlib import Path

root = Path("../dataset/selfplay")
cards = {c["slug"] for c in json.load(open("../assets/cards.json")) if c["type"] != "Tower Troop"}
plays, offsets, frames, complete, total, verified = collections.Counter(), [], [], 0, 0, [0, 0]
for m in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
    meta = json.load(open(m / "meta.json"))
    total += 1
    if not meta.get("complete"):
        continue
    complete += 1
    if meta.get("clock_offset_ms") is not None:
        offsets.append(meta["clock_offset_ms"])
    frames.append(len(list((m / "frames").glob("*.jpg"))))
    for line in open(m / "plays.jsonl"):
        p = json.loads(line)
        verified[0] += p["verified"]
        verified[1] += 1
        if p["verified"]:
            plays[p["card"]] += 1
low = sorted((plays[c], c) for c in cards if plays[c] < 50)
lines = [
    f"matches: {complete}/{total} complete",
    f"plays: {verified[1]} ({verified[0] / max(1, verified[1]):.1%} verified)",
    f"plays per card: min {min((plays[c] for c in cards), default=0)}, median {statistics.median([plays[c] for c in cards])}",
    f"cards under 50 plays ({len(low)}): " + ", ".join(f"{c} {n}" for n, c in low),
    f"clock offset ms: median {statistics.median(offsets) if offsets else None}, "
    f"range {min(offsets, default=None)}..{max(offsets, default=None)}",
    f"frames per match: median {statistics.median(frames) if frames else 0}",
]
text = "\n".join(lines)
print(text)
(root / "report.md").write_text("# Self-play dataset\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
