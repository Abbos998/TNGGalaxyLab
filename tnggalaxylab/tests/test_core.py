"""Tests for core modules."""

from __future__ import annotations

import h5py
import numpy as np

from tnggalaxylab.core.center import Centering
from tnggalaxylab.core.io import TNGCutoutReader
from tnggalaxylab.core.orientation import Orientation


def test_cutout_reader(tmp_path) -> None:
    path = tmp_path / "cutout.hdf5"
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["Time"] = 1.0
        stars = handle.create_group("PartType4")
        stars["Coordinates"] = np.ones((3, 3))
        stars["Velocities"] = np.zeros((3, 3))
        stars["Masses"] = np.arange(1.0, 4.0)
    data = TNGCutoutReader(path).load(components=("stars",))
    assert data.stars is not None
    assert data.stars.coordinates.shape == (3, 3)
    assert data.metadata["Header"]["Time"] == 1.0


def test_centering_methods() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0, 0.01])
    center = Centering.center_of_mass(positions, masses)
    assert center[0] < 2.0
    assert np.allclose(Centering.potential_minimum(positions, [0.0, -1.0, 1.0]), positions[1])
    shrink = Centering.shrinking_sphere(positions, masses, min_particles=2)
    assert shrink.shape == (3,)


def test_orientation() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    velocities = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    angular_momentum = Orientation.angular_momentum(positions, velocities)
    assert np.allclose(angular_momentum, [0.0, 0.0, 2.0])
    matrix = Orientation.face_on_matrix(angular_momentum)
    assert np.allclose(matrix @ angular_momentum, [0.0, 0.0, 2.0])
