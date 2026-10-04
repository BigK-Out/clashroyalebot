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


def test_positive_clip_uses_the_logged_play_not_the_proposal():
    # A proposal matched by chance 2 tiles and 1.2 s away must not move the labeled clip.
    props = [Proposal(2200, 6, 12, "motion")]
    plays = [{"card": "fireball", "t_view": 1000, "col": 4, "row": 10}]
    ev = label_events(props, plays, video_ms=60_000, lag_ms=500)
    assert ev == [{"t_ms": 1500, "col": 4, "row": 10, "label": "fireball", "proposed": True}]


def test_negatives_are_capped_and_positives_kept():
    from selfplay.clips import cap_negatives
    ev = [{"label": "knight"}] * 5 + [{"label": "no_play", "t_ms": i} for i in range(100)]
    out = cap_negatives(ev, ratio=2, seed=1)
    assert sum(e["label"] == "knight" for e in out) == 5
    assert sum(e["label"] == "no_play" for e in out) == 10
    assert out == cap_negatives(ev, ratio=2, seed=1)
