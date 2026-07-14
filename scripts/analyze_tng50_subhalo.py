"""
analyze_tng50_subhalo.py  (Stage 4 final)
==========================================
TNG50 cutout faylidan yulduzli diskni o'qib, Fourier moduli orqali
to'liq, publication-grade tahlil qiladi.

Stage 4 dagi ilmiy o'zgarishlar:
    CRITICAL fix A.1 — Velocity sqrt(a) unit correction (Springel 2010)
    CRITICAL fix A.2 — Three labelled rotation curves
    MAJOR     fix    — Phase-coherent bar length (Aguerri+2005)
    MAJOR     fix    — Iterative shrinking-sphere centering (Power et al. 2003)
    MAJOR     fix    — Optional Sigma(R) for rigorous Jog 2002 average
    MAJOR     fix    — Bootstrap default 100 -> 500 (Efron-Tibshirani 1993)

Foydalanish:
    python analyze_tng50_subhalo.py <cutout_29.hdf5> [output_dir]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# Stage 2/3/4 fourier moduli
from tnggalaxylab.fourier import (
    compute_fourier,
    global_lopsidedness,
    compute_pattern_diagnostics,
    bootstrap_fourier,
    plot_amplitude_profiles,
    plot_phase_profiles,
    plot_method_comparison,
    export_csv,
    export_json,
    rotation_curve_tracer,
    rotation_curve_spherical,
    rotation_curve_cylindrical,
    compare_rotation_curves,
)

# Stage 4: use existing Power-2003 shrinking sphere from core module
from tnggalaxylab.core.center import Centering


# ---------------------------------------------------------------------------
# 1. TNG cutout yuklash + birliklarni fizik qiymatlarga aylantirish
# ---------------------------------------------------------------------------

def load_tng_stars(path: str) -> dict:
    """
    Load TNG cutout stellar particles, applying Stage 3 CRITICAL fix A.1:

        v_phys [km/s] = V_code [km*sqrt(a)/s] * sqrt(a)

    Ref: Springel 2010 sec.3.3; Nelson et al. 2015 sec.2.4;
    IllustrisTNG data release docs (Particle data -> Velocities).
    """
    with h5py.File(path, "r") as f:
        h = float(f["Header"].attrs["HubbleParam"])
        a = float(f["Header"].attrs["Time"])
        redshift = float(f["Header"].attrs["Redshift"])
        snap = int(f["Header"].attrs["SnapshotNumber"])
        sim  = f["Header"].attrs["SimulationName"]
        if isinstance(sim, bytes):
            sim = sim.decode()

        stars = f["PartType4"]
        coords     = stars["Coordinates"][:].astype(np.float64)
        masses     = stars["Masses"][:].astype(np.float64)
        velocities = stars["Velocities"][:].astype(np.float64)
        potential  = stars["Potential"][:].astype(np.float64)
        form_time  = stars["GFM_StellarFormationTime"][:].astype(np.float64)

    # Wind filter: GFM_StellarFormationTime <= 0 -> wind particle
    stellar_mask = form_time > 0
    coords     = coords[stellar_mask]
    masses     = masses[stellar_mask]
    velocities = velocities[stellar_mask]
    potential  = potential[stellar_mask]

    velocities_kms = velocities * np.sqrt(a)         # km/s
    coords_kpc     = coords * a / h                  # kpc
    masses_msun    = masses * 1e10 / h               # M_sun

    print(f"\n[Yuklandi] {sim}, snapshot {snap}, z = {redshift:.3f}")
    print(f"  Yulduz zarralari: {len(coords_kpc):,} (wind chiqarib tashlandi)")
    print(f"  Umumiy yulduz massasi: {masses_msun.sum():.3e} M_sun")
    print(f"  HubbleParam h = {h:.4f},  a = {a:.4f}")

    return dict(
        pos=coords_kpc, mass=masses_msun, vel=velocities_kms,
        potential=potential, h=h, a=a, redshift=redshift,
        snapshot=snap, simulation=sim,
    )


# ---------------------------------------------------------------------------
# 2. Markazlash — Power et al. 2003 shrinking sphere
# ---------------------------------------------------------------------------

def find_center_power2003(pos, potential, mass, initial_radius=30.0):
    """Iterative shrinking-sphere centring per Power et al. 2003 MNRAS 338, 14."""
    seed = Centering.potential_minimum(pos, potential)
    pos_seeded = pos - seed
    center_rel = Centering.shrinking_sphere(
        pos_seeded, masses=mass, initial_radius=initial_radius,
        shrink_factor=0.8, min_particles=128,
        tolerance=1e-4, max_iterations=100,
    )
    return seed + center_rel


# ---------------------------------------------------------------------------
# 3. Face-on alignment via angular-momentum vector
# ---------------------------------------------------------------------------

def align_disk_face_on(pos, vel, mass, r_align=10.0):
    r = np.linalg.norm(pos, axis=1)
    inner = r < r_align
    L = np.sum(np.cross(pos[inner], vel[inner]) * mass[inner, None], axis=0)
    L_hat = L / np.linalg.norm(L)
    z_hat = np.array([0.0, 0.0, 1.0])
    v = np.cross(L_hat, z_hat)
    s = np.linalg.norm(v)
    c = np.dot(L_hat, z_hat)
    if s < 1e-10:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[ 0,    -v[2],  v[1]],
                       [ v[2],   0,   -v[0]],
                       [-v[1],  v[0],   0  ]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    return pos @ R.T, vel @ R.T, R


# ---------------------------------------------------------------------------
# 4. Disk scale length + surface density profile
# ---------------------------------------------------------------------------

def fit_scale_length_and_sigma(pos_xy, mass, r_bins, z_cut=3.0, pos_full=None):
    """Fit R_d and compute Sigma(R) on the Fourier bin centres."""
    if pos_full is not None:
        mask = np.abs(pos_full[:, 2]) < z_cut
        pos_xy = pos_xy[mask]
        mass   = mass[mask]

    R = np.sqrt(pos_xy[:, 0]**2 + pos_xy[:, 1]**2)
    dr = r_bins[1] - r_bins[0] if len(r_bins) > 1 else 0.1
    edges = np.concatenate([[r_bins[0] - dr/2], r_bins + dr/2])
    sigma = np.zeros_like(r_bins)
    for i in range(len(r_bins)):
        m = (R >= edges[i]) & (R < edges[i + 1])
        area = np.pi * (edges[i + 1]**2 - edges[i]**2)
        if area > 0:
            sigma[i] = mass[m].sum() / area

    fit_mask = (r_bins > 1.0) & (r_bins < 10.0) & (sigma > 0)
    if fit_mask.sum() >= 5:
        p = np.polyfit(r_bins[fit_mask], np.log(sigma[fit_mask]), 1)
        R_d = -1.0 / p[0]
    else:
        R_d = 3.0
    return float(R_d), sigma


# ---------------------------------------------------------------------------
# 5. Old (Stage 2) bar length for comparison
# ---------------------------------------------------------------------------

def _old_bar_length(profile, r_in, r_out, bar_threshold=0.2):
    """OLD (Stage 2) definition: outermost R where A_2 > threshold."""
    r = profile.r_bins
    mask = (r >= r_in) & (r <= r_out)
    A2 = profile.A(2)[mask]
    r_sel = r[mask]
    bm = A2 > bar_threshold
    return float(r_sel[bm][-1]) if bm.any() else 0.0


# ---------------------------------------------------------------------------
# 6. Pipeline
# ---------------------------------------------------------------------------

def analyze(cutout_path: str, output_dir: str = "outputs"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # (1) Load
    data = load_tng_stars(cutout_path)
    pos, vel, mass = data["pos"], data["vel"], data["mass"]

    # (2) Centre (Power 2003)
    print(f"\n[Markaz - Power et al. 2003 shrinking sphere]")
    center = find_center_power2003(pos, data["potential"], mass)
    pos -= center
    vel -= np.average(vel, axis=0, weights=mass)
    print(f"  Markaz: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) kpc")

    # (3) Face-on
    pos_rot, vel_rot, _ = align_disk_face_on(pos, vel, mass, r_align=10.0)
    print(f"[Aylantirish] Disk z-o'qiga moslashtirildi (L-vector method)")

    # (4) Disk particles
    x, y, z = pos_rot[:, 0], pos_rot[:, 1], pos_rot[:, 2]
    R_cyl = np.sqrt(x**2 + y**2)
    disk_mask = (np.abs(z) < 3.0) & (R_cyl < 30.0)
    x_d, y_d, m_d = x[disk_mask], y[disk_mask], mass[disk_mask]
    print(f"[Disk tanlandi] |z|<3 kpc, R<30 kpc: {disk_mask.sum():,} zarra")

    # (5) Fourier profile + R_d + Sigma(R)
    r_out_phys = 14.0
    profile = compute_fourier(
        x_d, y_d, m_d, method="particles",
        r_in=0.0, r_out=r_out_phys, n_bins=40, m_max=4,
    )
    R_d, sigma_R = fit_scale_length_and_sigma(
        np.column_stack([x_d, y_d]), m_d, profile.r_bins,
        z_cut=3.0, pos_full=pos_rot[disk_mask],
    )
    print(f"[Scale length] R_d = {R_d:.2f} kpc")
    print(f"[Fourier] R: 0-{r_out_phys:.1f} kpc, 40 bin, m_max=4 (particle method)")

    # (6) FFT for comparison
    profile_fft = compute_fourier(
        x_d, y_d, m_d, method="fft",
        r_in=0.0, r_out=r_out_phys, n_bins=40, m_max=4,
        n_pix=512, sigma_smooth=2.0,
    )

    # (7) Global lopsidedness — three variants for comparison
    gm_default = global_lopsidedness(profile, scale_length=R_d)
    gm_jog = global_lopsidedness(profile, scale_length=R_d,
                                  surface_density=sigma_R)
    print(f"\n[Global Lopsidedness - apertura {gm_default.r_range_kpc[0]:.2f}-"
          f"{gm_default.r_range_kpc[1]:.2f} kpc]")
    print(f"  R&Z95 literature average:    A_1 = {gm_default.A1_literature:.4f}")
    print(f"  Area-weighted (old default): A_1 = {gm_default.A1_integral:.4f}")
    print(f"  Jog 2002 with Sigma(R) NEW:  A_1 = {gm_jog.A1_integral:.4f}")
    delta_pct = 100*(gm_jog.A1_integral - gm_default.A1_integral)/max(gm_default.A1_integral, 1e-10)
    print(f"  A_1 (Jog) - A_1 (area) = {(gm_jog.A1_integral - gm_default.A1_integral):+.4f} "
          f"({delta_pct:+.1f}%)")

    # (8) Pattern diagnostics — old vs new bar length
    diag = compute_pattern_diagnostics(profile, bar_threshold=0.2)
    old_bl = _old_bar_length(profile, 0.0, r_out_phys, bar_threshold=0.2)
    print(f"\n[Pattern diagnostikasi - old vs new]")
    print(f"  Dominant mode:           m = {diag.dominant_mode}")
    print(f"  Bar length (OLD Stage 2): {old_bl:.2f} kpc"
          f"  (A_2 > 0.2 chegarasidagi eng tashqi R)")
    print(f"  Bar length (NEW Aguerri+2005 phase-coherent): {diag.bar_length:.2f} kpc")
    print(f"  Bar angle:               {np.degrees(diag.bar_angle):.2f} deg")
    print(f"  Pattern coherence:       {diag.pattern_coherence:.3f}")
    print(f"  Phase scatter m=2:       {np.degrees(diag.phase_scatter_m2):.1f} deg")
    if diag.bar_length == 0 and old_bl > 0:
        print(f"  --> NEW algoritm bar yo'qligini ANIQ rad etdi (phase incoherent)")
    elif abs(old_bl - diag.bar_length) > 0.5:
        print(f"  --> Bar length {(old_bl-diag.bar_length):+.2f} kpc o'zgardi")

    # (9) Bootstrap (n=500)
    print(f"\n[Bootstrap - 500 iteratsiya (Efron-Tibshirani 1993)]")
    boot = bootstrap_fourier(
        x_d, y_d, m_d, method="particles",
        n_bootstrap=500, seed=42,
        r_in=0.0, r_out=r_out_phys, n_bins=40, m_max=4,
        scale_length=R_d,
    )
    print(f"  A_1 = {boot['global_A1_mean']:.4f} +/- {boot['global_A1_std']:.4f}")
    print(f"  A_2 = {boot['global_A2_mean']:.4f} +/- {boot['global_A2_std']:.4f}")
    print(f"  L_bar = {boot['bar_length_mean']:.2f} +/- {boot['bar_length_std']:.2f} kpc")

    # (10) Plots
    plot_amplitude_profiles(
        boot["profile"], modes=(1, 2, 3, 4),
        r_range=gm_default.r_range_kpc, global_modes=gm_default,
        savepath=str(out / "fig_amplitudes.png"),
    )
    plot_phase_profiles(
        boot["profile"], modes=(1, 2),
        savepath=str(out / "fig_phases.png"),
    )
    plot_method_comparison(
        profile_fft, profile, m=1,
        savepath=str(out / "fig_method_comparison_m1.png"),
    )
    plot_method_comparison(
        profile_fft, profile, m=2,
        savepath=str(out / "fig_method_comparison_m2.png"),
    )

    # (11) Export
    export_csv(boot["profile"], global_modes=gm_default,
               outpath=str(out / "fourier_profile.csv"))
    export_json(boot["profile"], global_modes=gm_default, pattern_diag=diag,
                outpath=str(out / "fourier_summary.json"))

    # (12) Rotation curves — three methods
    print(f"\n[Rotation curves - 3 metod (Stage 3 CRITICAL fix A.2)]")
    pos_disk = pos_rot[disk_mask]
    vel_disk = vel_rot[disk_mask]
    rc_tracer = rotation_curve_tracer(
        pos_disk[:, 0], pos_disk[:, 1], pos_disk[:, 2],
        vel_disk[:, 0], vel_disk[:, 1], vel_disk[:, 2],
        m_d, r_in=0.5, r_out=min(8*R_d, 20.0), n_bins=25, z_max=3.0,
    )
    rc_sph = rotation_curve_spherical(
        pos_disk[:, 0], pos_disk[:, 1], pos_disk[:, 2],
        m_d, r_in=0.5, r_out=min(8*R_d, 20.0), n_bins=25,
    )
    rc_cyl = rotation_curve_cylindrical(
        pos_disk[:, 0], pos_disk[:, 1], pos_disk[:, 2],
        m_d, r_in=0.5, r_out=min(8*R_d, 20.0), n_bins=25, z_max=3.0,
    )
    cmp_rc = compare_rotation_curves(rc_tracer, rc_sph, rc_cyl)
    for name, info in cmp_rc.items():
        print(f"  {name:13s}: v_max = {info['v_max']:6.1f} km/s  "
              f"at R = {info['r_at_vmax']:5.2f} kpc")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for rc, color, label in zip(
        [rc_tracer, rc_cyl, rc_sph],
        ["C0", "C1", "C2"],
        ["Tracer (median v_phi) - physical",
         "Cylindrical M(<R)",
         "Spherical M(<R) - biased for disks"]
    ):
        finite = np.isfinite(rc.v_circ)
        ax.plot(rc.r_bins[finite], rc.v_circ[finite],
                color=color, label=label, linewidth=1.6)
        if rc.v_err is not None:
            err = rc.v_err[finite]
            ax.fill_between(rc.r_bins[finite],
                            rc.v_circ[finite] - err,
                            rc.v_circ[finite] + err,
                            color=color, alpha=0.2)
    ax.axvline(R_d, color="gray", linestyle=":", label=f"R_d = {R_d:.2f} kpc")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("v_c [km/s]")
    ax.set_title("Rotation curve - 3 method comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(out / "fig_rotation_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Grafiklar saqlandi] {out}/")
    print(f"[Eksport] CSV + JSON saqlandi")

    return dict(
        profile=profile, profile_fft=profile_fft,
        global_modes=gm_default, global_modes_jog=gm_jog,
        diagnostics=diag, bootstrap=boot,
        old_bar_length=old_bl,
        rotation_curve_tracer=rc_tracer,
        rotation_curve_spherical=rc_sph,
        rotation_curve_cylindrical=rc_cyl,
        R_d=R_d, surface_density=sigma_R,
        center=center, n_disk_particles=int(disk_mask.sum()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TNGGalaxyLab - Stage 4 (CRITICAL+MAJOR fixes integrated)"
    )
    parser.add_argument("cutout", help="TNG cutout HDF5 fayl yo'li")
    parser.add_argument("output_dir", nargs="?", default="outputs",
                        help="Natijalar uchun papka (default: outputs)")
    args = parser.parse_args()

    if not os.path.isfile(args.cutout):
        print(f"XATO: fayl topilmadi: {args.cutout}", file=sys.stderr)
        sys.exit(1)

    analyze(args.cutout, args.output_dir)
