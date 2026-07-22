"""Galaxy orientation and projection utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tnggalaxylab.utils.coordinates import as_position_array, rotate_positions


class Orientation:
    """Compute angular-momentum orientations and rotate galaxies."""

    @staticmethod
    def angular_momentum(
        positions: ArrayLike,
        velocities: ArrayLike,
        masses: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Compute total angular momentum.

        Args:
            positions: Particle coordinates relative to center.
            velocities: Particle velocities relative to systemic velocity.
            masses: Optional particle masses.

        Returns:
            Angular momentum vector.
        """

        pos = as_position_array(positions)
        vel = as_position_array(velocities)
        if pos.shape != vel.shape:
            raise ValueError("positions and velocities must have equal shape")
        specific = np.cross(pos, vel)
        if masses is None:
            return np.sum(specific, axis=0)
        weights = np.asarray(masses, dtype=np.float64)
        if weights.shape != (pos.shape[0],):
            raise ValueError("masses must have shape (N,)")
        return np.sum(specific * weights[:, None], axis=0)

    @staticmethod
    def rotation_matrix_from_vectors(
        source: ArrayLike,
        target: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return a rotation matrix mapping ``source`` onto ``target``.

        Args:
            source: Source vector.
            target: Target vector.

        Returns:
            Rotation matrix.
        """

        a = np.asarray(source, dtype=np.float64)
        b = np.asarray(target, dtype=np.float64)
        if a.shape != (3,) or b.shape != (3,):
            raise ValueError("source and target must be three-vectors")
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            raise ValueError("source and target must be non-zero")
        a = a / a_norm
        b = b / b_norm
        cross = np.cross(a, b)
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if np.allclose(cross, 0.0):
            if dot > 0.0:
                return np.eye(3)
            axis = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            cross = np.cross(a, axis)
            cross = cross / np.linalg.norm(cross)
            return -np.eye(3) + 2.0 * np.outer(cross, cross)

        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ]
        )
        return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / np.dot(cross, cross))

    @staticmethod
    def face_on_matrix(angular_momentum: ArrayLike) -> NDArray[np.float64]:
        """Return a matrix rotating angular momentum to the z-axis."""

        return Orientation.rotation_matrix_from_vectors(angular_momentum, [0.0, 0.0, 1.0])

    @staticmethod
    def edge_on_matrix(
        angular_momentum: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return a matrix rotating angular momentum to the y-axis."""

        return Orientation.rotation_matrix_from_vectors(
            angular_momentum,
            [0.0, 1.0, 0.0],
        )

    @staticmethod
    def rotate_velocities(
        velocities: ArrayLike,
        matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Rotate particle velocities with the same matrix as positions.

        Args:
            velocities: Particle velocities with shape ``(N, 3)``.
            matrix: Rotation matrix with shape ``(3, 3)``.

        Returns:
            Rotated velocity array.
        """

        return rotate_positions(velocities, matrix)

    @staticmethod
    def orient(
        positions: ArrayLike,
        velocities: ArrayLike | None = None,
        masses: ArrayLike | None = None,
        mode: str = "face-on",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Rotate coordinates into a face-on or edge-on frame.

        Args:
            positions: Particle coordinates.
            velocities: Particle velocities. Required unless angular momentum
                is supplied via positions in a custom workflow.
            masses: Optional particle masses.
            mode: Either ``"face-on"`` or ``"edge-on"``.

        Returns:
            Rotated positions, rotated velocities, and the rotation matrix.
        """

        if velocities is None:
            raise ValueError("velocities are required to infer orientation")
        angular_momentum = Orientation.angular_momentum(positions, velocities, masses)
        if mode == "face-on":
            matrix = Orientation.face_on_matrix(angular_momentum)
        elif mode == "edge-on":
            matrix = Orientation.edge_on_matrix(angular_momentum)
        else:
            raise ValueError("mode must be 'face-on' or 'edge-on'")
        rotated_positions = rotate_positions(positions, matrix)
        rotated_velocities = Orientation.rotate_velocities(velocities, matrix)
        return (
            rotated_positions,
            rotated_velocities,
            matrix,
        )
