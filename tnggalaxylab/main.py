"""Command-line entry point for TNGGalaxyLab Fourier analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from tnggalaxylab.analysis.fourier_fft import FourierAnalyzer
from tnggalaxylab.core.io import TNGCutoutReader
from tnggalaxylab.core.orientation import Orientation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run TNGGalaxyLab Fourier analysis on a cutout.")
    parser.add_argument("cutout", type=Path, help="Path to an IllustrisTNG-style HDF5 cutout.")
    parser.add_argument("--output-dir", type=Path, default=Path("TNGGalaxyLab/output"))
    parser.add_argument("--label", default="galaxy")
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--n-radial", type=int, default=128)
    parser.add_argument("--n-azimuth", type=int, default=256)
    parser.add_argument("--max-mode", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Run the configured analysis."""

    args = parse_args()
    data = TNGCutoutReader(args.cutout).load(components=("stars",))
    if data.stars is None:
        raise SystemExit("No stellar component found in cutout.")
    positions = data.stars.coordinates
    velocities = data.stars.velocities
    masses = data.stars.masses
    if velocities is not None:
        positions, velocities, rotation_matrix = Orientation.orient(
            positions=positions,
            velocities=velocities,
            masses=masses,
            mode="face-on",
        )
    analyzer = FourierAnalyzer(
        positions,
        masses,
        max_mode=args.max_mode,
        output_dir=args.output_dir,
        label=args.label,
    )
    report = analyzer.report(
        bins=args.bins,
        n_radial=args.n_radial,
        n_azimuth=args.n_azimuth,
    )
    print(f"CSV: {report['csv']}")
    print(f"NPZ: {report['npz']}")
    for key, value in report["diagnostics"].items():
        print(f"{key}: {value:.6g}")


if __name__ == "__main__":
    main()
