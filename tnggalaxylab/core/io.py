"""Input/output routines for particle cutouts and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray


PARTICLE_GROUPS = {
    "gas": "PartType0",
    "dark_matter": "PartType1",
    "stars": "PartType4",
    "black_holes": "PartType5",
}


@dataclass(slots=True)
class ParticleComponent:
    """Container for one particle component.

    Attributes:
        coordinates: Particle coordinates in simulation units.
        velocities: Particle velocities, if available.
        masses: Particle masses, if available.
        extra: Additional datasets keyed by dataset name.
    """

    coordinates: NDArray[np.float64]
    velocities: NDArray[np.float64] | None = None
    masses: NDArray[np.float64] | None = None
    extra: dict[str, NDArray[Any]] = field(default_factory=dict)


@dataclass(slots=True)
class GalaxyData:
    """Loaded galaxy particle data.

    Attributes:
        stars: Stellar particles.
        gas: Gas particles.
        dark_matter: Dark matter particles.
        black_holes: Black hole particles.
        metadata: Header and file-level metadata.
    """

    stars: ParticleComponent | None = None
    gas: ParticleComponent | None = None
    dark_matter: ParticleComponent | None = None
    black_holes: ParticleComponent | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def available_components(self) -> list[str]:
        """Return names of loaded particle components."""

        return [
            name
            for name in ("stars", "gas", "dark_matter", "black_holes")
            if getattr(self, name) is not None
        ]


class TNGCutoutReader:
    """Read IllustrisTNG-style HDF5 cutouts.

    The reader accepts normal TNG group names (``PartType0`` etc.) and is
    deliberately permissive about missing datasets so it can also ingest many
    EAGLE/Gadget/RAMSES HDF5 exports after light conversion.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the reader.

        Args:
            path: HDF5 cutout path.
        """

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def load(
        self,
        components: tuple[str, ...] = ("stars", "gas", "dark_matter", "black_holes"),
        extra_fields: tuple[str, ...] = (),
    ) -> GalaxyData:
        """Load selected particle components.

        Args:
            components: Component names to load.
            extra_fields: Additional dataset names copied when present.

        Returns:
            GalaxyData instance.
        """

        invalid = set(components) - set(PARTICLE_GROUPS)
        if invalid:
            raise ValueError(f"unknown component(s): {sorted(invalid)}")

        with h5py.File(self.path, "r") as handle:
            metadata = self._read_metadata(handle)
            loaded: dict[str, ParticleComponent | None] = {}
            for component in components:
                loaded[component] = self._read_component(
                    handle,
                    PARTICLE_GROUPS[component],
                    extra_fields=extra_fields,
                )

        return GalaxyData(
            stars=loaded.get("stars"),
            gas=loaded.get("gas"),
            dark_matter=loaded.get("dark_matter"),
            black_holes=loaded.get("black_holes"),
            metadata=metadata,
        )

    def _read_metadata(self, handle: h5py.File) -> dict[str, Any]:
        metadata: dict[str, Any] = {"path": str(self.path)}
        for group_name in ("Header", "Config", "Parameters"):
            if group_name in handle:
                metadata[group_name] = {
                    key: self._decode_attr(value)
                    for key, value in handle[group_name].attrs.items()
                }
        metadata["root_attrs"] = {
            key: self._decode_attr(value) for key, value in handle.attrs.items()
        }
        return metadata

    def _read_component(
        self,
        handle: h5py.File,
        group_name: str,
        extra_fields: tuple[str, ...],
    ) -> ParticleComponent | None:
        if group_name not in handle:
            return None

        group = handle[group_name]
        coordinates = self._dataset(group, ("Coordinates", "coordinates", "pos"))
        if coordinates is None:
            return None

        velocities = self._dataset(group, ("Velocities", "Velocity", "velocities", "vel"))
        masses = self._dataset(group, ("Masses", "Mass", "masses", "mass"))
        extra = {
            field: np.asarray(group[field])
            for field in extra_fields
            if field in group
        }
        return ParticleComponent(
            coordinates=np.asarray(coordinates, dtype=np.float64),
            velocities=None if velocities is None else np.asarray(velocities, dtype=np.float64),
            masses=None if masses is None else np.asarray(masses, dtype=np.float64),
            extra=extra,
        )

    @staticmethod
    def _dataset(group: h5py.Group, names: tuple[str, ...]) -> NDArray[Any] | None:
        for name in names:
            if name in group:
                return np.asarray(group[name])
        return None

    @staticmethod
    def _decode_attr(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
