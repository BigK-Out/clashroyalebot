"""Deck inference (spec section 3): the opponent has 8 cards and a 4-card hand cycle."""
import math

import numpy as np

BEAM, TOP_K, SURE, DECK, CYCLE = 64, 10, 0.98, 8, 3


def refine(probs, no_play):
    beams = [((), (), 0.0, [])]  # (cards seen, last plays, logp, path)
    for p in probs:
        order = [k for k in np.argsort(-p) if k != no_play][:TOP_K]
        nxt = []
        for seen, last, lp, path in beams:
            for k in order:
                k = int(k)
                sure = p[k] >= SURE
                new_seen = seen if k in seen else seen + (k,)
                if not sure and (len(new_seen) > DECK or k in last):
                    continue
                nxt.append((new_seen, (last + (k,))[-CYCLE:], lp + math.log(max(p[k], 1e-12)), path + [k]))
        if not nxt:  # every option broke a rule: fall back to the plain argmax
            k = int(order[0])
            nxt = [(s, (l + (k,))[-CYCLE:], lp, path + [k]) for s, l, lp, path in beams]
        beams = sorted(nxt, key=lambda b: -b[2])[:BEAM]
    return np.array(beams[0][3])
