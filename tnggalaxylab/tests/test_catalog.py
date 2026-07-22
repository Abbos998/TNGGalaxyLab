"""Tests for catalog processing."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from tnggalaxylab.catalog.batch import CatalogProcessor


def test_catalog_processor(tmp_path) -> None:
    path = tmp_path / "galaxy.hdf5"
    rng = np.random.default_rng(4)
    positions = rng.normal(0.0, 1.0, size=(1000, 3))
    with h5py.File(path, "w") as handle:
        stars = handle.create_group("PartType4")
        stars["Coordinates"] = positions
        stars["Masses"] = np.ones(positions.shape[0])
    catalog = pd.DataFrame({"galaxy_id": ["g1"], "path": [path]})
    table = CatalogProcessor(catalog, output_dir=tmp_path / "out").run_fourier(
        workers=1,
        analyzer_kwargs={"max_mode": 4},
    )
    assert table.shape[0] == 1
    assert "bar_length" in table.columns
