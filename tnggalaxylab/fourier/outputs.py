"""
tnggalaxylab.fourier.outputs
=============================
Publication-quality plots and structured data export for Fourier results.

All plot functions return Matplotlib Figure objects and optionally save
to disk.  No global state is modified.

CSV and JSON exports are structured for direct ingestion by downstream
analysis pipelines.
"""

from __future__ import annotations

import json
import csv
import os
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from numpy.typing import NDArray

from .core import FourierProfile, GlobalModes


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_STYLE = dict(linewidth=1.5)

def _apply_style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=11)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Radial amplitude profiles
# ---------------------------------------------------------------------------

def plot_amplitude_profiles(
    profile: FourierProfile,
    modes: Sequence[int] = (1, 2, 3, 4),
    r_range: Optional[tuple] = None,
    global_modes: Optional[GlobalModes] = None,
    savepath: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot radial A_m(R) profiles for selected modes.

    Parameters
    ----------
    profile : FourierProfile
    modes : sequence of int   Modes to plot (1-indexed).
    r_range : (R_in, R_out) or None
        If given, draws vertical shading over the lopsidedness aperture.
    global_modes : GlobalModes or None
        If given, annotates the global A_1 value.
    savepath : str or None    If given, save the figure here.

    Returns
    -------
    matplotlib.figure.Figure
    """
    modes = [m for m in modes if 1 <= m <= profile.m_max]
    fig, ax = plt.subplots(figsize=figsize)

    for mi, m in enumerate(modes):
        A_m = profile.A(m)
        color = _COLORS[mi % len(_COLORS)]
        label = f"$A_{m}$"
        ax.plot(profile.r_bins, A_m, color=color, label=label, **_STYLE)
        if profile.amp_err is not None:
            err = profile.amp_err[:, m - 1]
            ax.fill_between(
                profile.r_bins, A_m - err, A_m + err,
                color=color, alpha=0.2,
            )

    # Aperture shading
    if r_range is not None:
        ax.axvspan(*r_range, color="gray", alpha=0.12, label=f"Aperture [{r_range[0]:.1f}, {r_range[1]:.1f}] kpc")

    # Annotate global A_1
    if global_modes is not None:
        ax.axhline(global_modes.A1_literature, color=_COLORS[0],
                   linestyle="--", linewidth=1.2,
                   label=f"$\\langle A_1 \\rangle$ = {global_modes.A1_literature:.3f}")

    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=10, framealpha=0.8)
    _apply_style(ax,
                 xlabel="$R$ [kpc]",
                 ylabel="Fourier amplitude $A_m$",
                 title=f"Radial Fourier profiles  [method: {profile.method}]")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Phase profiles
# ---------------------------------------------------------------------------

def plot_phase_profiles(
    profile: FourierProfile,
    modes: Sequence[int] = (1, 2),
    savepath: Optional[str] = None,
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """
    Plot radial phase Phi_m(R) profiles.

    Parameters
    ----------
    profile : FourierProfile
    modes : sequence of int   Modes to plot.
    savepath : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    modes = [m for m in modes if 1 <= m <= profile.m_max]
    fig, ax = plt.subplots(figsize=figsize)

    for mi, m in enumerate(modes):
        phi_m = np.degrees(profile.Phi(m))  # plot in degrees
        color = _COLORS[mi % len(_COLORS)]
        ax.scatter(profile.r_bins, phi_m, s=12, color=color,
                   label=f"$\\Phi_{m}$")
        if profile.phase_err is not None:
            err_deg = np.degrees(profile.phase_err[:, m - 1])
            ax.errorbar(
                profile.r_bins, phi_m, yerr=err_deg,
                fmt="none", ecolor=color, alpha=0.4, capsize=2,
            )

    ax.set_ylim(-180, 180)
    ax.set_yticks(np.arange(-180, 181, 45))
    ax.legend(fontsize=10, framealpha=0.8)
    _apply_style(ax,
                 xlabel="$R$ [kpc]",
                 ylabel="Phase $\\Phi_m$ [deg]",
                 title=f"Fourier phase profiles  [method: {profile.method}]")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Method comparison plot
# ---------------------------------------------------------------------------

def plot_method_comparison(
    profile_fft: FourierProfile,
    profile_par: FourierProfile,
    m: int = 1,
    savepath: Optional[str] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Side-by-side comparison of FFT vs particle Fourier for mode *m*.

    Parameters
    ----------
    profile_fft : FourierProfile   method="fft"
    profile_par : FourierProfile   method="particles"
    m : int                        Mode to compare.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, prof, label, color in zip(
        axes,
        [profile_fft, profile_par],
        ["FFT", "Particles"],
        [_COLORS[0], _COLORS[1]],
    ):
        A_m = prof.A(m)
        ax.plot(prof.r_bins, A_m, color=color, label=label, **_STYLE)
        if prof.amp_err is not None:
            err = prof.amp_err[:, m - 1]
            ax.fill_between(prof.r_bins, A_m - err, A_m + err,
                            color=color, alpha=0.2)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=10)
        _apply_style(ax,
                     xlabel="$R$ [kpc]",
                     ylabel=f"$A_{m}$",
                     title=f"$A_{m}(R)$ вЂ” {label}")

    fig.suptitle(f"Method comparison: mode $m={m}$", fontsize=12, y=1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Synthetic validation plot
# ---------------------------------------------------------------------------

def plot_validation(
    profile: FourierProfile,
    analytic_amps: dict,
    m: int = 1,
    savepath: Optional[str] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Compare recovered vs analytic Fourier amplitude.

    Parameters
    ----------
    profile : FourierProfile
        Numerical result.
    analytic_amps : dict
        {m: callable(R) -> A_m(R)} from a SyntheticGalaxy.
    m : int
        Mode to validate.
    """
    fig, ax = plt.subplots(figsize=figsize)

    A_rec = profile.A(m)
    A_ana = analytic_amps[m](profile.r_bins)

    ax.plot(profile.r_bins, A_rec, color=_COLORS[0],
            label=f"Recovered $A_{m}$  [{profile.method}]", **_STYLE)
    ax.plot(profile.r_bins, A_ana, color="black",
            linestyle="--", linewidth=1.5, label=f"Analytic $A_{m}$")

    if profile.amp_err is not None:
        err = profile.amp_err[:, m - 1]
        ax.fill_between(profile.r_bins, A_rec - err, A_rec + err,
                        color=_COLORS[0], alpha=0.2, label="Bootstrap 1Пѓ")

    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=10)
    _apply_style(ax,
                 xlabel="$R$ [kpc]",
                 ylabel=f"$A_{m}$",
                 title=f"Validation: mode $m={m}$  [{profile.method}]")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Summary exports
# ---------------------------------------------------------------------------

def export_csv(
    profile: FourierProfile,
    global_modes: Optional[GlobalModes] = None,
    outpath: str = "fourier_summary.csv",
) -> None:
    """
    Export radial Fourier profiles to CSV.

    Columns: r_kpc, A1, A2, ..., Am, Phi1_deg, ..., Phim_deg,
             [A1_err, A2_err, ..., Phi1_err_deg, ...] if bootstrap present.
    A header comment line records global A1/A2 if provided.
    """
    rows = []
    header = ["r_kpc"]
    for m in range(1, profile.m_max + 1):
        header.append(f"A{m}")
    for m in range(1, profile.m_max + 1):
        header.append(f"Phi{m}_deg")
    if profile.amp_err is not None:
        for m in range(1, profile.m_max + 1):
            header.append(f"A{m}_err")
        for m in range(1, profile.m_max + 1):
            header.append(f"Phi{m}_err_deg")

    for i, r in enumerate(profile.r_bins):
        row = [f"{r:.6f}"]
        for m in range(1, profile.m_max + 1):
            row.append(f"{profile.A(m)[i]:.8f}")
        for m in range(1, profile.m_max + 1):
            row.append(f"{np.degrees(profile.Phi(m)[i]):.6f}")
        if profile.amp_err is not None:
            for m in range(1, profile.m_max + 1):
                row.append(f"{profile.amp_err[i, m-1]:.8f}")
            for m in range(1, profile.m_max + 1):
                row.append(f"{np.degrees(profile.phase_err[i, m-1]):.6f}")
        rows.append(row)

    with open(outpath, "w", newline="") as f:
        if global_modes is not None:
            f.write(f"# global_A1_literature={global_modes.A1_literature:.6f}"
                    f"  global_A2_literature={global_modes.A2_literature:.6f}"
                    f"  aperture_kpc={global_modes.r_range_kpc}\n")
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def export_json(
    profile: FourierProfile,
    global_modes: Optional[GlobalModes] = None,
    pattern_diag=None,
    outpath: str = "fourier_summary.json",
) -> None:
    """
    Export full Fourier results to JSON.

    Suitable for archiving alongside publication figures.
    """
    def _f(arr):
        return [round(float(v), 8) for v in arr]

    data = {
        "method": profile.method,
        "r_in_kpc": profile.r_in,
        "r_out_kpc": profile.r_out,
        "n_bins": profile.n_bins,
        "m_max": profile.m_max,
        "r_bins_kpc": _f(profile.r_bins),
    }

    for m in range(1, profile.m_max + 1):
        data[f"A{m}"] = _f(profile.A(m))
        data[f"Phi{m}_deg"] = _f(np.degrees(profile.Phi(m)))
    if profile.amp_err is not None:
        for m in range(1, profile.m_max + 1):
            data[f"A{m}_err"] = _f(profile.amp_err[:, m - 1])
            data[f"Phi{m}_err_deg"] = _f(np.degrees(profile.phase_err[:, m - 1]))

    if global_modes is not None:
        data["global"] = {
            "A1_literature": round(global_modes.A1_literature, 6),
            "A2_literature": round(global_modes.A2_literature, 6),
            "A1_integral":   round(global_modes.A1_integral,   6),
            "A2_integral":   round(global_modes.A2_integral,   6),
            "r_range_kpc": list(global_modes.r_range_kpc),
            "scale_length_kpc": global_modes.scale_length_kpc,
        }

    if pattern_diag is not None:
        data["pattern"] = {
            "dominant_mode": pattern_diag.dominant_mode,
            "pattern_angle_deg": round(np.degrees(pattern_diag.pattern_angle), 3),
            "bar_angle_deg": round(np.degrees(pattern_diag.bar_angle), 3)
                             if not np.isnan(pattern_diag.bar_angle) else None,
            "bar_length_kpc": round(pattern_diag.bar_length, 4),
            "pattern_coherence": round(pattern_diag.pattern_coherence, 4)
                                  if not np.isnan(pattern_diag.pattern_coherence) else None,
            "phase_scatter_m1_deg": round(np.degrees(pattern_diag.phase_scatter_m1), 3),
            "phase_scatter_m2_deg": round(np.degrees(pattern_diag.phase_scatter_m2), 3)
                                    if not np.isnan(pattern_diag.phase_scatter_m2) else None,
        }

    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
