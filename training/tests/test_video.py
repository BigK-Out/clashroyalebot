import numpy as np

from selfplay.video import match_offset

COST = {"knight": 3, "zap": 2}


def series_with_drops(drops, length_ms=60_000, start=7.0):
    """Elixir at 30 fps: regenerates slowly, drops by `cost` at each (t, cost)."""
    t = np.arange(0, length_ms, 33.0)
    v = np.full_like(t, start)
    for td, cost in drops:
        v[t >= td] -= cost
    return np.stack([t, v], 1)


def test_constant_offset_found():
    start = 1_000_000
    plays = [{"host_ms": start + 5_000 * i + 2_000, "card": "knight", "verified": True} for i in range(6)]
    drops = [(p["host_ms"] - start - 1_460, 3) for p in plays]
    off = match_offset(series_with_drops(drops), plays, start, COST)
    assert off is not None
    offset, n, spread = off
    assert abs(offset - (-1_460)) <= 40 and n == 6 and spread <= 80


def test_too_few_matches_gives_none():
    start = 0
    plays = [{"host_ms": 10_000, "card": "knight", "verified": True}]
    assert match_offset(series_with_drops([(8_540, 3)]), plays, start, COST) is None


def test_drop_smaller_than_cost_is_not_a_match():
    start = 0
    plays = [{"host_ms": 10_000 + 5_000 * i, "card": "knight", "verified": True} for i in range(4)]
    drops = [(p["host_ms"] - 1_460, 1) for p in plays]  # only 1 elixir drops: not this card
    assert match_offset(series_with_drops(drops), plays, start, COST) is None


def test_frequent_plays_do_not_steal_each_others_drops():
    # A play every 1.5 s: the wide first search window also holds the previous play's drop.
    start = 0
    plays = [{"host_ms": 10_000 + 1_500 * i, "card": "zap" if i % 2 else "knight", "verified": True} for i in range(12)]
    drops = [(p["host_ms"] - 1_460, COST[p["card"]]) for p in plays]
    offset, n, spread = match_offset(series_with_drops(drops, start=30.0), plays, start, COST)
    assert abs(offset - (-1_460)) <= 40 and spread <= 80, (offset, spread)


def test_second_pass_ignores_unrelated_earlier_drops():
    # Some plays have an unrelated bigger drop ~0.9 s before theirs (e.g. the bar flashing).
    start = 0
    plays = [{"host_ms": 10_000 + 5_000 * i, "card": "knight", "verified": True} for i in range(10)]
    drops = [(p["host_ms"] - 1_460, 3) for p in plays]
    drops += [(p["host_ms"] - 2_360, 3) for p in plays[:4]]
    offset, n, spread = match_offset(series_with_drops(drops, start=40.0), plays, start, COST)
    assert abs(offset - (-1_460)) <= 40 and spread <= 80, (offset, spread)
