"""Clip dataset for every aligned match that has none yet -> ../dataset/selfplay_clips/."""
from pathlib import Path

from selfplay.clips import build_match

out = Path("../dataset/selfplay_clips")
for m in sorted(Path("../dataset/selfplay").iterdir()):
    if (m / "align.json").exists() and not (out / f"{m.name}_note14.npz").exists():
        build_match(m, out)
