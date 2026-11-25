"""The per-length walk layout, shared by every script in this folder.

GPM's multiscale mask gives a walk its length by position: the first quarter of
the K walks is read at 2 hops, the next at 4, then 6, then 8. In the original
nested runs the walks of a stage are appended after the earlier ones, so most
old walks move to a shorter quarter when K grows, and their encodings change.

Here the walks added at each stage are split into one equal chunk per length,
and a walk keeps that length for good:

    stage   new walks   per length after the stage
    K=16    w0-w15      4    (w0-w3 at 2 hops, w4-w7 at 4, w8-w11 at 6, w12-w15 at 8)
    K=32    w16-w31     8    (w16-w19 at 2 hops, w20-w23 at 4, ...)
    K=64    w32-w63     16
    K=128   w64-w127    32

At any K the walks are placed length group by length group, so GPM's own mask
assigns each group its length and no model code changes. K=16 is the same as
the original layout. Walk numbers are positions in sampling order: the
prediction stage's 128-walk pool, or the live run's stages appended in turn.
"""
import numpy as np

STEPS = (16, 32, 64, 128)


def check_steps(steps, groups):
    previous = 0
    for k in steps:
        if k <= previous or (k - previous) % groups:
            raise ValueError(f'Each stage must add a positive multiple of {groups} walks; got {steps}.')
        previous = k


def order(k, steps=STEPS, groups=4):
    """Positions (in sampling order) of the walks used at budget k, grouped by length.

    k need not be a stage: fixed K=48 takes the K=32 walks plus the first 4 of
    each length's chunk from the K=64 stage, giving 12 walks per length.
    """
    check_steps(steps, groups)
    if k < 1 or k > steps[-1] or k % groups:
        raise ValueError(f'k must be a multiple of {groups} between {groups} and {steps[-1]}; got {k}.')
    out = [[] for _ in range(groups)]
    previous = 0
    for stage in steps:
        if previous >= k:
            break
        chunk = (stage - previous) // groups
        take = (min(stage, k) - previous) // groups
        for g in range(groups):
            start = previous + g * chunk
            out[g].extend(range(start, start + take))
        previous = stage
    return np.array([i for group in out for i in group], dtype=np.int64)


def lengths(k, multiscale):
    """Walk length GPM's mask gives each position of order(k), for checks and tests."""
    return np.repeat(np.array(multiscale), k // len(multiscale))


def split_new(count, groups=4):
    """Slices of a stage's new walks, one per length group, in sampling order."""
    if count % groups:
        raise ValueError(f'{count} new walks cannot be split into {groups} equal groups.')
    size = count // groups
    return [slice(g * size, (g + 1) * size) for g in range(groups)]
