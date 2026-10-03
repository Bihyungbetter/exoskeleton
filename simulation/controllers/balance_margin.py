"""Extrapolated centre of mass (Hof) and its margin to the base of support.

    XCoM = CoM_xy + v_xy / omega0,    omega0 = sqrt(g / h_CoM)

The margin is the signed distance from XCoM to the convex hull of the active
foot-floor contacts, positive inside. Negative means a step is needed.

The CoM is the whole-model one (subtree_com[0]), so it includes the exo mass;
h_CoM uses the same combined CoM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81


@dataclass(frozen=True)
class BalanceMargin:
    xcom: np.ndarray            # extrapolated CoM, world xy
    margin_m: float             # signed distance to the support polygon, + inside
    omega0: float               # sqrt(g / h_com)
    support_points: int         # contacts forming the polygon
    valid: bool                 # False when there is no support to measure against


def _signed_distance_to_hull(point: np.ndarray, pts: np.ndarray) -> float:
    """Signed distance from ``point`` to the convex hull of ``pts`` (2-D).

    Positive inside. One or two contacts have no interior, so the result is
    minus the distance to that point or segment.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return float("nan")
    if len(pts) == 1:
        return -float(np.linalg.norm(point - pts[0]))
    if len(pts) == 2:
        return -_distance_to_segment(point, pts[0], pts[1])
    vertices = _hull_vertices_ccw(pts)
    if len(vertices) < 3:
        # Collinear support: use the segment along whichever axis has the
        # larger extent (x only would collapse a support segment along y).
        axis = int(np.argmax(np.ptp(pts, axis=0)))
        lo, hi = pts[np.argmin(pts[:, axis])], pts[np.argmax(pts[:, axis])]
        return -_distance_to_segment(point, lo, hi)

    inside = True
    best = float("inf")
    n = len(vertices)
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        edge = b - a
        # Vertices are CCW, so an inside point is left of every edge.
        # 2-D cross written out because np.cross on 2-D vectors is deprecated.
        d = point - a
        if float(edge[0] * d[1] - edge[1] * d[0]) < 0.0:
            inside = False
        best = min(best, _distance_to_segment(point, a, b))
    return float(best if inside else -best)


def _hull_vertices_ccw(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D ``pts`` as counter-clockwise vertices.

    Andrew's monotone chain. Gives the same vertex set as scipy's ConvexHull
    (strict corners only) without qhull opening a temp file on every call,
    which is slow on Windows. Returns fewer than three vertices for collinear
    or non-finite input so the caller falls back to a segment.
    """
    if not np.all(np.isfinite(pts)):
        return pts[:0]                 # caller falls back to a segment
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = [(float(pts[i, 0]), float(pts[i, 1])) for i in order]

    def turn(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for q in p:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], q) <= 0.0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in reversed(p):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], q) <= 0.0:
            upper.pop()
        upper.append(q)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return pts[:0]
    return np.asarray(hull, dtype=float)


def _distance_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def compute_balance_margin(measurement) -> BalanceMargin:
    """Return the XCoM and its signed margin to the current base of support."""
    com = np.asarray(measurement.com_position, dtype=float)
    vel = np.asarray(measurement.com_velocity, dtype=float)
    height = max(float(com[2]), 1e-3)
    omega0 = float(np.sqrt(GRAVITY / height))
    xcom = com[:2] + vel[:2] / omega0

    pts = np.asarray(measurement.contact_points, dtype=float)
    if len(pts) == 0:
        return BalanceMargin(xcom, float("nan"), omega0, 0, False)
    margin = _signed_distance_to_hull(xcom, pts[:, :2])
    return BalanceMargin(xcom, float(margin), omega0, int(len(pts)), True)
