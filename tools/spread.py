"""Fire-spread solver: turns a fuel map into a burn-arrival map.

Pure Python + numpy so it runs both in the system Python (stand-in maps)
and inside Blender's bundled Python (the real bake).

Model
-----
The front advances normal to itself at the local fuel speed, pushed by an
upward draught (fire climbs a sheet faster than it creeps down). That is a
shortest-travel-time problem, solved with Dijkstra over a 16-neighbour
stencil: a 4- or 8-neighbour stencil would give octagonal fronts.

Spot fires (embers landing ahead of the front) are added as extra sources
with a head start, so small holes open ahead of the main front and merge.
"""
import heapq
import math

import numpy as np

# 8 king moves + 8 knight moves: worst-case direction error ~2.7%.
STENCIL = [(dx, dy) for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)
           if (dx, dy) != (0, 0) and math.gcd(abs(dx), abs(dy)) == 1]


def _dijkstra(slow, sources, updraught):
    """slow: (n, n) slowness (1/speed). sources: [(row, col, t0)].
    updraught k in [0, 1): speed is (1+k) straight up, (1-k) straight down."""
    n = slow.shape[0]
    pad = 2
    m = n + 2 * pad
    big = float("inf")
    s = np.full((m, m), big, dtype=np.float64)
    s[pad:-pad, pad:-pad] = slow
    S = s.ravel().tolist()
    T = [big] * (m * m)
    done = [False] * (m * m)

    k = updraught
    moves = []
    for dx, dy in STENCIL:
        length = math.hypot(dx, dy)
        cos_up = -dy / length  # image rows grow downward, so "up" is -y
        # travel time for a unit step at unit speed with the draught (Zermelo)
        g = (-k * cos_up + math.sqrt(k * k * cos_up * cos_up + 1 - k * k)) / (1 - k * k)
        moves.append((dy * m + dx, 0.5 * length * g))

    heap = []
    for r, c, t0 in sources:
        i = (r + pad) * m + (c + pad)
        if t0 < T[i]:
            T[i] = t0
            heap.append((t0, i))
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop

    while heap:
        t, i = pop(heap)
        if done[i]:
            continue
        done[i] = True
        si = S[i]
        for off, c in moves:
            j = i + off
            sj = S[j]
            if sj == big or done[j]:
                continue
            nt = t + c * (si + sj)
            if nt < T[j]:
                T[j] = nt
                push(heap, (nt, j))

    out = np.array(T, dtype=np.float64).reshape(m, m)[pad:-pad, pad:-pad]
    return out


def solve(fuel, updraught=0.28, spots=7, seed=3):
    """fuel: (n, n) speed multiplier, ~0.3 (damp) .. 1.6 (dry).
    Returns arrival time T normalised to 0..1 and |grad T| per texel."""
    n = fuel.shape[0]
    slow = 1.0 / np.clip(fuel, 0.12, 4.0)
    c = n // 2
    # a small dry patch at the ignition point so the burn always takes
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - c, yy - c)
    slow = slow * (1 - 0.6 * np.exp(-(r / (n * 0.03)) ** 2))

    main = _dijkstra(slow, [(c, c, 0.0)], updraught)

    sources = [(c, c, 0.0)]
    rng = np.random.default_rng(seed)
    tries = 0
    while len(sources) < spots + 1 and tries < 400:
        tries += 1
        ang = rng.uniform(-math.pi, 0) if rng.random() < 0.75 else rng.uniform(0, math.pi)
        rad = rng.uniform(0.16, 0.62) * n * 0.5
        sx = int(c + math.cos(ang) * rad)
        sy = int(c + math.sin(ang) * rad)
        if not (4 <= sx < n - 4 and 4 <= sy < n - 4):
            continue
        if any(math.hypot(sx - q[1], sy - q[0]) < n * 0.09 for q in sources):
            continue
        # an ember lands shortly before the main front would have arrived
        sources.append((sy, sx, main[sy, sx] * rng.uniform(0.74, 0.9)))

    T = _dijkstra(slow, sources, updraught) if len(sources) > 1 else main

    # knock back the faint stencil faceting
    T = _blur3(_blur3(T))
    T = T - T.min()
    T = T / T.max()
    gy, gx = np.gradient(T)
    grad = _blur3(np.hypot(gx, gy))
    return T.astype(np.float32), grad.astype(np.float32)


def _blur3(a):
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:] +
            2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:] +
            p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) / 16.0


def pack(T, grad):
    """Pack to 8-bit RGB: R,G = 16-bit arrival (hi, lo), B = |grad T|.
    Returns (rgb uint8 array, meta dict the page needs to decode it)."""
    v = np.clip(np.round(T * 65535), 0, 65535).astype(np.uint32)
    gmax = float(np.percentile(grad, 99.7))
    b = np.clip(np.round(grad / gmax * 255), 1, 255).astype(np.uint8)
    rgb = np.dstack([(v >> 8).astype(np.uint8), (v & 255).astype(np.uint8), b])
    return rgb, {"n": int(T.shape[0]), "gradMax": gmax}
