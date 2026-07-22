"""
============================================================
TNGGalaxyLab

Rotation Curve Analysis

Version 1.2

Author : Abbos Omonov
============================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass(slots=True)
class RotationCurveResult:

    radius: np.ndarray

    velocity: np.ndarray

    sigma: np.ndarray

    counts: np.ndarray


class RotationCurveAnalyzer:

    """
    Stellar rotation curve.
    """

    def __init__(
        self,
        positions,
        velocities,
        masses=None,
    ):

        self.positions = np.asarray(positions)

        self.velocities = np.asarray(velocities)

        self.masses = masses

        self.result = None

    # =====================================================

    def compute(
        self,
        bins=50,
    ):

        x = self.positions[:, 0]

        y = self.positions[:, 1]

        vx = self.velocities[:, 0]

        vy = self.velocities[:, 1]

        r = np.sqrt(x**2 + y**2)

        good = r > 0

        r = r[good]

        x = x[good]

        y = y[good]

        vx = vx[good]

        vy = vy[good]

        vphi = (x * vy - y * vx) / r

        edges = np.linspace(
            0,
            r.max(),
            bins + 1,
        )

        rc = 0.5 * (edges[:-1] + edges[1:])

        vel = np.zeros(bins)

        sig = np.zeros(bins)

        cnt = np.zeros(bins, dtype=int)

        for i in range(bins):

            m = (
                (r >= edges[i])
                &
                (r < edges[i + 1])
            )

            cnt[i] = np.sum(m)

            if cnt[i] == 0:

                vel[i] = np.nan

                sig[i] = np.nan

                continue

            vel[i] = np.mean(vphi[m])

            sig[i] = np.std(vphi[m])

        self.result = RotationCurveResult(

            radius=rc,

            velocity=vel,

            sigma=sig,

            counts=cnt,

        )

        return self.result