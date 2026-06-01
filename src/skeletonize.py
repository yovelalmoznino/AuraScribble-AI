"""
Pure-numpy skeletonization + stroke tracing.

Turns a binary glyph/word bitmap into pen-like CENTERLINE strokes (ordered
points), instead of the filled outline a font path gives or the raster scan an
image dump gives. Used by both the synthetic Hebrew generator and the HHD
image -> stroke converter so the model trains on something closer to real ink.

No scikit-image / scipy dependency: Zhang-Suen thinning and connected-component
labeling are implemented directly on numpy arrays.
"""

from __future__ import annotations

import numpy as np


def to_binary(arr: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Foreground (ink) = 1 where the grayscale value is darker than threshold."""
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return (arr < threshold).astype(np.uint8)


def zhang_suen_thin(binary: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """
    Zhang-Suen thinning -> 1px-wide skeleton. `binary` is a {0,1} uint8 array
    with 1 = ink. Vectorized over the whole image per sub-iteration.
    """
    img = (binary > 0).astype(np.uint8)
    if img.sum() == 0:
        return img

    def neighbors(p):
        # P2..P9 clockwise from north; p is the padded image.
        P2 = p[:-2, 1:-1]
        P3 = p[:-2, 2:]
        P4 = p[1:-1, 2:]
        P5 = p[2:, 2:]
        P6 = p[2:, 1:-1]
        P7 = p[2:, :-2]
        P8 = p[1:-1, :-2]
        P9 = p[:-2, :-2]
        return P2, P3, P4, P5, P6, P7, P8, P9

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1, mode="constant")
            P2, P3, P4, P5, P6, P7, P8, P9 = neighbors(p)
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            # A = number of 0->1 transitions in the ordered neighbor sequence.
            A = np.zeros(img.shape, dtype=np.uint8)
            for a, b in zip(seq[:-1], seq[1:]):
                A += ((a == 0) & (b == 1)).astype(np.uint8)
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

            cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if step == 0:
                cond &= (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond &= (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)

            if cond.any():
                img[cond] = 0
                changed = True
        if not changed:
            break
    return img


def _label_components(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected components as lists of (y, x) pixels. Iterative BFS."""
    visited = np.zeros_like(skel, dtype=bool)
    h, w = skel.shape
    comps: list[list[tuple[int, int]]] = []
    ys, xs = np.where(skel > 0)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        visited[sy, sx] = True
        comp: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        comps.append(comp)
    return comps


def _trace_component(pixels: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Greedy nearest-neighbor walk through a component's pixels, starting at the
    rightmost (then topmost) pixel so Hebrew traces right-to-left like real
    writing. Bridges small gaps; returns an ordered list of (x, y).
    """
    remaining = set(pixels)
    # Start: max x, then min y.
    start = max(pixels, key=lambda p: (p[1], -p[0]))
    order: list[tuple[int, int]] = [start]
    remaining.discard(start)
    cur = start
    while remaining:
        cy, cx = cur
        # nearest by squared distance
        best = None
        best_d = None
        for (py, px) in remaining:
            d = (py - cy) ** 2 + (px - cx) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best = (py, px)
        cur = best
        remaining.discard(best)
        order.append(best)
    # return as (x, y)
    return [(x, y) for (y, x) in order]


def skeleton_to_strokes(
    binary: np.ndarray,
    *,
    max_points: int = 220,
    rtl: bool = True,
    gap_jump: float = 1.0,
) -> list[list[float]]:
    """
    Full pipeline: thin -> components -> ordered trace -> [x, y, t] points.

    Components are ordered right-to-left (rtl=True) by their rightmost x so that
    Hebrew words come out in natural writing order. A pen-up gap (extra time
    step) is inserted between components. y is kept as image-y (down-positive),
    matching the app's screen convention.
    """
    skel = zhang_suen_thin(binary)
    comps = _label_components(skel)
    if not comps:
        return []

    # Order components: RTL -> by descending max x; else ascending min x.
    def comp_key(comp: list[tuple[int, int]]):
        max_x = max(px for (_, px) in comp)
        min_x = min(px for (_, px) in comp)
        return -max_x if rtl else min_x

    comps.sort(key=comp_key)

    points: list[list[float]] = []
    t = 0.0
    for comp in comps:
        if len(comp) < 2:
            continue
        traced = _trace_component(comp)
        for i, (x, y) in enumerate(traced):
            points.append([float(x), float(y), t])
            # pen-up gap at the start of a new stroke
            t += (1.0 + gap_jump) if i == 0 and points else 1.0
        t += gap_jump  # gap after the stroke

    if len(points) < 4:
        return []

    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = [points[i] for i in idx]

    return points
