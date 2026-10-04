"""Write align.json for every complete --both match that has none yet."""
import json
from pathlib import Path

from selfplay.video import align_match

for m in sorted(Path("../dataset/selfplay").iterdir()):
    meta = m / "meta.json"
    if not (m / "note9.mkv").exists() or not meta.exists() or not json.load(open(meta)).get("complete"):
        continue
    if (m / "align.json").exists():
        continue
    print(m.name, align_match(m), flush=True)
