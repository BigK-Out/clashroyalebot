import numpy as np

from selfplay.proposals import Proposal, match_props, merge, motion_proposals, tag_proposals


def test_new_tag_is_a_proposal_and_a_tracked_one_is_not():
    frames = [(0, []), (100, [[0.5, 0.3, 9, 8]]), (200, [[0.5, 0.31, 9, 8]]), (300, [[0.5, 0.32, 9, 9]])]
    props = tag_proposals(frames)
    assert [(p.t_ms, p.col, p.row) for p in props] == [(100, 9, 8)]


def test_flicker_does_not_make_a_new_proposal():
    frames = [(0, [[0.5, 0.3, 9, 8]]), (100, []), (200, [[0.5, 0.3, 9, 8]])]
    assert len(tag_proposals(frames)) == 1


def test_motion_burst_in_quiet_area():
    quiet = np.zeros((32, 18))
    burst = quiet.copy()
    burst[5:8, 4:7] = 0.9
    diffs = [(0, quiet), (100, quiet), (200, quiet), (300, burst)]
    props = motion_proposals(diffs)
    assert len(props) == 1 and props[0].kind == "motion" and (props[0].col, props[0].row) == (5, 6)


def test_merge_keeps_earliest_of_close_proposals():
    p = [Proposal(1000, 5, 5, "motion"), Proposal(1200, 6, 5, "tag"), Proposal(5000, 5, 5, "tag")]
    assert merge(p) == [Proposal(1000, 5, 5, "motion"), Proposal(5000, 5, 5, "tag")]


def test_match_props_by_time_and_place():
    props = [Proposal(1300, 4, 10, "tag"), Proposal(9000, 4, 10, "tag")]
    plays = [{"t_view": 1000, "col": 4, "row": 11}, {"t_view": 20000, "col": 4, "row": 11}]
    pairs, unmatched = match_props(props, plays)
    assert pairs == {0: 0} and unmatched == [1]
