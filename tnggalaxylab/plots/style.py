"""Publication plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_publication_style() -> None:
    """Apply a clean plotting style suitable for journal figures."""

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.8,
        }
    )


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    tight: bool = True,
) -> Path:
    """Save and close a Matplotlib figure.

    Args:
        fig: Figure to save.
        path: Output path.
        tight: Whether to use tight bounding boxes.

    Returns:
        Resolved output path.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight" if tight else None)
    plt.close(fig)
    return output
