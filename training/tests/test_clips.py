import numpy as np

from selfplay.clips import CLASSES, crop, label_events
from selfplay.proposals import Proposal


def test_classes_are_122_cards_plus_no_play():
    assert len(CLASSES) == 123 and CLASSES[-1] == "no_play" and "hog_rider" in CLASSES and "tower_princess" not in CLASSES


def test_crop_at_corner_is_padded():
    f = np.full((1280, 576, 3), 200, np.uint8)
    c = crop(f, 0, 0, 180)
    assert c.shape == (128, 128, 3)
    assert c[:60, :60].max() == 0 and c[-10:, -10:].min() == 200


def test_unmatched_proposal_is_no_play():
    props = [Proposal(1300, 4, 10, "tag"), Proposal(9000, 8, 20, "tag")]
    plays = [{"card": "knight", "t_view": 1000, "col": 4, "row": 11}]
    ev = label_events(props, plays, video_ms=60_000)
    assert [(e["label"], e["proposed"]) for e in ev] == [("knight", True), ("no_play", True)]


def test_missed_play_is_added_unproposed():
    plays = [{"card": "zap", "t_view": 5000, "col": 9, "row": 8}]
    ev = label_events([], plays, video_ms=60_000)
    assert ev == [{"t_ms": 5300, "col": 9, "row": 8, "label": "zap", "proposed": False}]


def test_plays_outside_video_are_skipped():
    plays = [{"card": "zap", "t_view": -2000, "col": 9, "row": 8}, {"card": "zap", "t_view": 59_500, "col": 9, "row": 8}]
    assert label_events([], plays, video_ms=60_000) == []
