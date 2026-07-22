"""Bar diagnostics based on Fourier mode profiles."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class BarDiagnostics:
    """Measure bar phase stability, angle, length, and pattern speed."""

    def __init__(
        self,
        radius: ArrayLike,
        a2_over_a0: ArrayLike,
        phase2: ArrayLike,
    ) -> None:
        """Initialize from m=2 amplitude and phase radial profiles."""

        self.radius = np.asarray(radius, dtype=np.float64)
        self.a2_over_a0 = np.asarray(a2_over_a0, dtype=np.float64)
        self.phase2 = np.unwrap(np.asarray(phase2, dtype=np.float64))
        if not (self.radius.shape == self.a2_over_a0.shape == self.phase2.shape):
            raise ValueError("radius, a2_over_a0, and phase2 must have the same shape")

    def phase_stability(self, threshold: float = np.deg2rad(10.0)) -> np.ndarray:
        """Return mask where m=2 phase is stable around the peak."""

        if self.radius.size == 0:
            return np.array([], dtype=bool)
        peak = int(np.nanargmax(self.a2_over_a0))
        reference = self.phase2[peak]
        return np.abs(np.angle(np.exp(1j * (self.phase2 - reference)))) <= threshold

    def bar_angle(self) -> float:
        """Return the bar position angle in radians."""

        peak = int(np.nanargmax(self.a2_over_a0))
        return float(0.5 * self.phase2[peak])

    def bar_length(
        self,
        amplitude_fraction: float = 0.5,
        phase_threshold: float = np.deg2rad(10.0),
    ) -> float:
        """Estimate bar length from m=2 amplitude and phase stability."""

        if not 0.0 < amplitude_fraction <= 1.0:
            raise ValueError("amplitude_fraction must be in (0, 1]")
        peak = int(np.nanargmax(self.a2_over_a0))
        minimum = amplitude_fraction * self.a2_over_a0[peak]
        stable = self.phase_stability(phase_threshold)
        valid = stable & (self.a2_over_a0 >= minimum)
        candidates = np.where(valid & (np.arange(self.radius.size) >= peak))[0]
        if candidates.size == 0:
            return float(self.radius[peak])
        return float(self.radius[candidates[-1]])

    @staticmethod
    def pattern_speed(
        angles: ArrayLike,
        times: ArrayLike,
    ) -> float:
        """Estimate pattern speed from bar angle evolution.

        Args:
            angles: Bar angles in radians.
            times: Simulation times.

        Returns:
            Linear slope ``d angle / d time``.
        """

        angle = np.unwrap(np.asarray(angles, dtype=np.float64))
        time = np.asarray(times, dtype=np.float64)
        if angle.shape != time.shape or angle.size < 2:
            raise ValueError("angles and times must have equal size >= 2")
        slope, _ = np.polyfit(time, angle, 1)
        return float(slope)
