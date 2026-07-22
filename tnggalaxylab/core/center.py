"""Center-finding algorithms for galaxy particle distributions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tnggalaxylab.utils.coordinates import as_position_array


class Centering:
    """Collection of particle centering algorithms."""

    @staticmethod
    def center_of_mass(
        positions: ArrayLike,
        masses: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Compute the mass-weighted center of mass.

        Args:
            positions: Particle coordinates with shape ``(N, 3)``.
            masses: Optional particle masses.

        Returns:
            Three-vector center.
        """

        pos = as_position_array(positions)
        if masses is None:
            return np.mean(pos, axis=0)
        weights = np.asarray(masses, dtype=np.float64)
        if weights.shape != (pos.shape[0],):
            raise ValueError("masses must have shape (N,)")
        if not np.any(weights > 0.0):
            raise ValueError("at least one mass must be positive")
        return np.average(pos, axis=0, weights=weights)

    @staticmethod
    def potential_minimum(
        positions: ArrayLike,
        potentials: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return the coordinate of the particle with minimum potential.

        Args:
            positions: Particle coordinates with shape ``(N, 3)``.
            potentials: Particle gravitational potential values.

        Returns:
            Three-vector potential minimum position.
        """

        pos = as_position_array(positions)
        pot = np.asarray(potentials, dtype=np.float64)
        if pot.shape != (pos.shape[0],):
            raise ValueError("potentials must have shape (N,)")
        return pos[int(np.nanargmin(pot))].copy()

    @staticmethod
    def shrinking_sphere(
        positions: ArrayLike,
        masses: ArrayLike | None = None,
        initial_radius: float | None = None,
        shrink_factor: float = 0.8,
        min_particles: int = 128,
        tolerance: float = 1.0e-4,
        max_iterations: int = 100,
    ) -> NDArray[np.float64]:
        """Estimate the galaxy center with the shrinking-sphere method.

        Args:
            positions: Particle coordinates with shape ``(N, 3)``.
            masses: Optional particle masses.
            initial_radius: Starting sphere radius. Defaults to max radius.
            shrink_factor: Multiplicative radius shrinkage per iteration.
            min_particles: Stop once fewer particles would remain.
            tolerance: Stop when center displacement is below this value.
            max_iterations: Maximum iterations.

        Returns:
            Three-vector center estimate.
        """

        pos = as_position_array(positions)
        weights = None if masses is None else np.asarray(masses, dtype=np.float64)
        if weights is not None and weights.shape != (pos.shape[0],):
            raise ValueError("masses must have shape (N,)")
        center = Centering.center_of_mass(pos, weights)
        radius = (
            float(np.max(np.linalg.norm(pos - center, axis=1)))
            if initial_radius is None
            else float(initial_radius)
        )
        if radius <= 0.0:
            return center

        active = np.ones(pos.shape[0], dtype=bool)
        for _ in range(max_iterations):
            distances = np.linalg.norm(pos - center, axis=1)
            next_active = distances <= radius
            if np.count_nonzero(next_active) < min_particles:
                break
            new_center = Centering.center_of_mass(
                pos[next_active],
                None if weights is None else weights[next_active],
            )
            shift = float(np.linalg.norm(new_center - center))
            center = new_center
            active = next_active
            radius *= shrink_factor
            if shift < tolerance:
                break

        return Centering.center_of_mass(
            pos[active],
            None if weights is None else weights[active],
        )

    @staticmethod
    def iterative_centering(
        positions: ArrayLike,
        masses: ArrayLike | None = None,
        percentile: float = 50.0,
        max_iterations: int = 32,
        tolerance: float = 1.0e-4,
    ) -> NDArray[np.float64]:
        """Iteratively recenter on progressively tighter radial subsets.

        Args:
            positions: Particle coordinates with shape ``(N, 3)``.
            masses: Optional particle masses.
            percentile: Radius percentile kept each iteration.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance in coordinate units.

        Returns:
            Three-vector center estimate.
        """

        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be between 0 and 100")
        pos = as_position_array(positions)
        weights = None if masses is None else np.asarray(masses, dtype=np.float64)
        center = Centering.center_of_mass(pos, weights)
        active = np.ones(pos.shape[0], dtype=bool)
        for _ in range(max_iterations):
            distances = np.linalg.norm(pos[active] - center, axis=1)
            radius = np.percentile(distances, percentile)
            local = active.copy()
            local[active] = distances <= radius
            if np.count_nonzero(local) < 16:
                break
            new_center = Centering.center_of_mass(
                pos[local],
                None if weights is None else weights[local],
            )
            if np.linalg.norm(new_center - center) < tolerance:
                center = new_center
                break
            center = new_center
            active = local
        return center
