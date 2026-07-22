"""Catalog-scale processing tools."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from tnggalaxylab.analysis.fourier_fft import FourierAnalyzer
from tnggalaxylab.core.io import TNGCutoutReader


@dataclass(slots=True)
class CatalogTask:
    """Single-galaxy processing task."""

    galaxy_id: str
    path: str | Path


def _process_cutout(
    task: CatalogTask,
    output_dir: str | Path,
    analyzer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    reader = TNGCutoutReader(task.path)
    data = reader.load(components=("stars",))
    if data.stars is None:
        raise ValueError(f"{task.path} contains no stellar particle component")
    analyzer = FourierAnalyzer(
        data.stars.coordinates,
        data.stars.masses,
        output_dir=Path(output_dir) / str(task.galaxy_id),
        label=str(task.galaxy_id),
        **analyzer_kwargs,
    )
    report = analyzer.report(make_plots=False)
    row = {"galaxy_id": task.galaxy_id, "path": str(task.path)}
    row.update(report["global_modes"])
    row.update(report["diagnostics"])
    row["csv"] = str(report["csv"])
    row["npz"] = str(report["npz"])
    return row


class CatalogProcessor:
    """Batch-process simulation cutouts and write CSV summaries."""

    def __init__(
        self,
        tasks: list[CatalogTask] | pd.DataFrame,
        output_dir: str | Path = "TNGGalaxyLab/output/catalog",
    ) -> None:
        """Initialize a catalog processor.

        Args:
            tasks: Either a list of ``CatalogTask`` or a DataFrame with
                ``galaxy_id`` and ``path`` columns.
            output_dir: Directory for per-galaxy products and summaries.
        """

        if isinstance(tasks, pd.DataFrame):
            required = {"galaxy_id", "path"}
            missing = required - set(tasks.columns)
            if missing:
                raise ValueError(f"catalog DataFrame missing columns: {sorted(missing)}")
            self.tasks = [
                CatalogTask(str(row.galaxy_id), row.path)
                for row in tasks.itertuples(index=False)
            ]
        else:
            self.tasks = tasks
        self.output_dir = Path(output_dir)

    def run_fourier(
        self,
        workers: int = 1,
        summary_path: str | Path | None = None,
        analyzer_kwargs: dict[str, Any] | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> pd.DataFrame:
        """Run Fourier analysis for all catalog entries.

        Args:
            workers: Number of worker processes.
            summary_path: Optional CSV summary path.
            analyzer_kwargs: Keyword arguments passed to ``FourierAnalyzer``.
            callback: Optional callable invoked with each result row.

        Returns:
            Summary table.
        """

        kwargs = analyzer_kwargs or {}
        rows: list[dict[str, Any]] = []
        if workers <= 1:
            iterator = tqdm(self.tasks, desc="Processing galaxies")
            for task in iterator:
                row = _process_cutout(task, self.output_dir, kwargs)
                rows.append(row)
                if callback is not None:
                    callback(row)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_process_cutout, task, self.output_dir, kwargs)
                    for task in self.tasks
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing galaxies"):
                    row = future.result()
                    rows.append(row)
                    if callback is not None:
                        callback(row)

        table = pd.DataFrame(rows).sort_values("galaxy_id").reset_index(drop=True)
        output = Path(summary_path) if summary_path is not None else self.output_dir / "summary.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        return table
