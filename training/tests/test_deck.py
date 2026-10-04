import numpy as np

from selfplay.deck import refine


def onehot_mix(n_classes, best, second, p_best):
    p = np.full(n_classes, 1e-4)
    p[best], p[second] = p_best, 1 - p_best - 1e-4 * (n_classes - 2)
    return p


def test_cycle_flips_an_unsure_repeat():
    # Plays: 0, 1, then 0 again at 55% (impossible right after) vs 2 at 45%: pick 2.
    c = 10
    probs = np.stack([onehot_mix(c, 0, 1, 0.99), onehot_mix(c, 1, 0, 0.99), onehot_mix(c, 0, 2, 0.55)])
    assert list(refine(probs, no_play=c - 1)) == [0, 1, 2]


def test_confident_prediction_wins_over_cycle():
    c = 10
    probs = np.stack([onehot_mix(c, 0, 1, 0.99), onehot_mix(c, 0, 2, 0.99)])
    assert list(refine(probs, no_play=c - 1)) == [0, 0]


def test_ninth_distinct_card_is_replaced_by_a_known_one():
    c = 20
    rows = [onehot_mix(c, k, (k + 1) % 8, 0.99) for k in range(8)] * 2  # 8 cards, cycled
    rows.append(onehot_mix(c, 15, 3, 0.6))  # unsure 9th distinct card vs known card 3
    out = refine(np.stack(rows), no_play=c - 1)
    assert out[-1] == 3
