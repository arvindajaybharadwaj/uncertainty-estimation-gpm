"""Tests for the per-length layout and the pipeline commands. No GPU or data needed.

Run from the repository root:
    python -m pytest encoder_caching/test_encoder_caching.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402
import run_checkpoint  # noqa: E402

MULTISCALE = [2, 4, 6, 8]


def length_of(k):
    """Walk number -> length at budget k, as GPM's position mask assigns it."""
    return dict(zip(layout.order(k).tolist(), layout.lengths(k, MULTISCALE).tolist()))


def test_k16_is_the_original_layout():
    assert layout.order(16).tolist() == list(range(16))


def test_k32_groups_match_the_plan():
    assert layout.order(32).tolist() == [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23,
                                         8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31]


def test_fixed_48_takes_four_per_length_from_the_third_stage():
    two_hop = layout.order(48)[:12].tolist()
    assert two_hop == [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35]
    assert sorted(layout.order(48).tolist()) == sorted(
        list(range(32)) + [32 + 8 * g + j for g in range(4) for j in range(4)])


@pytest.mark.parametrize('k', [16, 32, 48, 64, 128])
def test_every_budget_uses_each_walk_once_with_equal_groups(k):
    o = layout.order(k)
    assert len(o) == k and len(set(o.tolist())) == k
    counts = np.unique(layout.lengths(k, MULTISCALE), return_counts=True)[1]
    assert (counts == k // 4).all()


def test_a_walk_keeps_its_length_as_k_grows():
    for small, large in [(16, 32), (32, 64), (64, 128), (32, 48), (48, 64)]:
        a, b = length_of(small), length_of(large)
        assert all(b[w] == a[w] for w in a), (small, large)


def test_budgets_are_nested():
    for small, large in [(16, 32), (32, 48), (48, 64), (64, 128)]:
        assert set(layout.order(small).tolist()) <= set(layout.order(large).tolist())


def test_live_split_matches_the_offline_order():
    """Appending each stage's split chunks to per-length caches reproduces layout.order."""
    cache, previous = [[] for _ in range(4)], 0
    for k in layout.STEPS:
        new = list(range(previous, k))
        for g, s in enumerate(layout.split_new(k - previous)):
            cache[g].extend(new[s])
        assert [w for group in cache for w in group] == layout.order(k).tolist()
        previous = k


def test_invalid_budgets_are_refused():
    for k in (0, 10, 130):
        with pytest.raises(ValueError):
            layout.order(k)
    with pytest.raises(ValueError):
        layout.split_new(10)


def test_pipeline_steps_and_smoke_check():
    steps = run_checkpoint.commands('wikics', 1, smoke=True)
    assert [s[0] for s in steps] == list(run_checkpoint.STEPS)
    assert '--check' in steps[1][2] and '--smoke' in steps[0][2]
    assert all('encoder_caching' in str(output) for _, output, _ in steps)
    assert '--check' not in run_checkpoint.commands('wikics', 1, smoke=False)[1][2]
