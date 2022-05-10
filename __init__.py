from .poibin import PoiBin
from .cpoibin import (
    find_threshold,
    find_threshold_scipy,
    set_threshold,
    find_threshold_dot,
    generate_wisdom,
    pdf,
    convolve_pdf,
)

# planner flags
FFTW_ESTIMATE = 64
FFTW_MEASURE = 0
FFTW_PATIENT = 32
FFTW_EXHAUSTIVE = 8
FFTW_WISDOM_ONLY = 2097152

