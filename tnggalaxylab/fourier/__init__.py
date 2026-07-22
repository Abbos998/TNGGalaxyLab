"""
tnggalaxylab.fourier
====================
Publication-quality Fourier decomposition of galaxy surface density,
plus literature-cited rotation-curve methods.

Public API:
    Fourier:         compute_fourier, fft_fourier, particle_fourier
                     global_lopsidedness, FourierProfile, GlobalModes
    Diagnostics:     compute_pattern_diagnostics, PatternDiagnostics
    Synthetic:       make_exponential_disk, make_lopsided_disk,
                     make_barred_disk, make_logarithmic_spiral,
                     SyntheticGalaxy
    Bootstrap:       bootstrap_fourier  (default n=500, Efron-Tibshirani 1993)
    Plots:           plot_amplitude_profiles, plot_phase_profiles,
                     plot_method_comparison, plot_validation
    Exports:         export_csv, export_json
    Kinematics:      rotation_curve_tracer,         (physical, median v_phi)
                     rotation_curve_spherical,      (M(<R)/R - labelled approximation)
                     rotation_curve_cylindrical,    (M(<R,|z|<h)/R - approximation)
                     compare_rotation_curves
                     RotationCurve, G_KPC_MSUN_KMS
"""
from .core import (
    compute_fourier, fft_fourier, particle_fourier,
    global_lopsidedness, FourierProfile, GlobalModes,
)
from .diagnostics import compute_pattern_diagnostics, PatternDiagnostics
from .synthetic import (
    make_exponential_disk, make_lopsided_disk,
    make_barred_disk, make_logarithmic_spiral, SyntheticGalaxy,
)
from .bootstrap import bootstrap_fourier
from .outputs import (
    plot_amplitude_profiles, plot_phase_profiles,
    plot_method_comparison, plot_validation,
    export_csv, export_json,
)
from .kinematics import (
    rotation_curve_tracer,
    rotation_curve_spherical,
    rotation_curve_cylindrical,
    compare_rotation_curves,
    RotationCurve,
    G_KPC_MSUN_KMS,
)
