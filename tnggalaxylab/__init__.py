"""Structural and dynamical galaxy analysis tools.

TNGGalaxyLab provides reusable, documented workflows for particle-based
galaxy analysis across IllustrisTNG, EAGLE, Gadget, RAMSES, and related
simulation outputs.
"""
__version__ = "0.1.0"
__all__ = []

# Eski submodullarni qulay yorliqlar sifatida import qilish (xato bo'lsa o'tkazib yuborish)
try:
    from tnggalaxylab.analysis.fourier_fft import FourierAnalyzer
    __all__.append("FourierAnalyzer")
except Exception:
    pass

try:
    from tnggalaxylab.core.io import GalaxyData, TNGCutoutReader
    __all__.extend(["GalaxyData", "TNGCutoutReader"])
except Exception:
    pass
