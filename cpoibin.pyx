# cython: profile=False
from libc.math cimport ceil, log2
cimport openmp
cimport fftw
from fftw cimport fftw_plan
cimport scipy.linalg.cython_blas as blas
import os, time, typing
import numpy as np
import scipy.signal as signal

# smallest size to use the fft instead of the direct method for convolution
# scipy threshold is 1025 to 4097 while fftw is much faster than naive convolve
cdef int THRESHOLD = 17

# smallest size to use blas ddot instead of a raw C loop
cdef int THRESHOLD_DOT = 33

# file to store wisdom
cdef char *wisdom_filename = "wisdom"
cdef char *wisdom_backup = "wisdom~"

# initialize threading
cdef int num_threads = openmp.omp_get_max_threads()
if num_threads > 1:
    fftw.fftw_init_threads()
    # maximum number of threads fftw is allowed to use
    fftw.fftw_plan_with_nthreads(num_threads)

# load fftw wisdom from system and local directory if either exists
fftw.fftw_import_system_wisdom()
fftw.fftw_import_wisdom_from_filename(wisdom_filename)
fftw.fftw_import_wisdom_from_filename(wisdom_backup)

### helper methods

cdef int half_ceil(int n):
    """ Return half of n, rounded up. """
    return (n + 1)//2

cdef int next_size(int n):
    """ Return the next size after the given size. """
    return 2*n - 1

cdef int size_space(int n):
    """ Return the space taken up by the given size. """
    return 2*(n - 1)

### user functions

# wisdom utility functions

def generate_wisdom(n: int, flags: int=fftw.FFTW_EXHAUSTIVE,
                    verbose: bool=True) -> None:
    """ Generate wisdom for all sizes up to n. """
    cdef:
        int size
        double *x
        fftw_plan plan_forward, plan_backward

    size = 1
    while size <= n:
        # use fftw malloc for proper alignment
        x = fftw.fftw_alloc_real(size)

        if verbose:
            print(f"Planning transform: r2hc{size}")

        plan_forward = \
            fftw.fftw_plan_r2r_1d(size, x, x, fftw.FFTW_R2HC, flags)

        if verbose:
            print(fftw.fftw_sprint_plan(plan_forward).decode("utf-8"))
            print(f"Planning transform: hc2r{size}")

        plan_backward = \
            fftw.fftw_plan_r2r_1d(size, x, x, fftw.FFTW_HC2R, flags)

        if verbose:
            print(fftw.fftw_sprint_plan(plan_backward).decode("utf-8"))

        # clean up
        fftw.fftw_destroy_plan(plan_forward)
        fftw.fftw_destroy_plan(plan_backward)
        fftw.fftw_free(x)

        # save wisdom to file, write to backup first in case of interrupt
        if fftw.fftw_export_wisdom_to_filename(wisdom_backup):
            if fftw.fftw_export_wisdom_to_filename(wisdom_filename):
                os.remove(wisdom_backup)

        size = 2*size

# threshold finding functions

def measure_time(function: typing.Callable[[], None], trials: int) -> float:
    """ Time a function over trials iterations. """
    start = time.time()
    for _ in range(trials):
        function()
    return (time.time() - start)/trials

def set_threshold(threshold: int) -> int:
    """ Set the threshold. """
    global THRESHOLD

    THRESHOLD = threshold

def find_threshold_scipy(measure: bool=False, verbose: bool=True) -> int:
    """ Use scipy to estimate the threshold to switch from direct to fft. """
    global THRESHOLD

    n = 2
    while True:
        x, y = np.empty(n), np.empty(n)
        ret = signal.choose_conv_method(x, y, measure=measure)
        method = ret if not measure else ret[0]
        if measure and verbose:
            direct_time, fft_time = ret[1]["direct"], ret[1]["fft"]
            print(f"Size {n:4}: direct {direct_time:.3e} fft {fft_time:.3e} "
                  f"ratio {direct_time/fft_time:.3f}")
        if method == "fft":
            THRESHOLD = n
            return THRESHOLD

        n = next_size(n)

def find_threshold(trials: int=1000, flags: int=fftw.FFTW_ESTIMATE,
                   verbose: bool=True) -> int:
    """ Time both methods to estimate the fft threshold. """
    cdef:
        int n, space
        double *x
        double *y
        double *temp
        fftw_plan plan_forward, plan_backward

    global THRESHOLD

    n = 3
    while True:
        space = size_space(n)
        # allocate both inputs at once
        x = fftw.fftw_alloc_real(2*space)
        y = x + space
        temp = fftw.fftw_alloc_real(space)

        plan_forward = \
            fftw.fftw_plan_r2r_1d(space, x, x, fftw.FFTW_R2HC, flags)
        plan_backward = \
            fftw.fftw_plan_r2r_1d(space, x, x, fftw.FFTW_HC2R, flags)

        direct_time = measure_time(lambda: direct_convolve(n, x, n, y), trials)
        fft_time = measure_time(
            lambda: fft_convolve(space, x, space, y,
                                 plan_forward, plan_backward, temp),
            trials
        )

        if verbose:
            print(f"Size {n:4}: direct {direct_time:.3e} fft {fft_time:.3e}"
                  f"ratio {direct_time/fft_time:.3f}")

        # clean up
        fftw.fftw_destroy_plan(plan_forward)
        fftw.fftw_destroy_plan(plan_backward)
        fftw.fftw_free(x)

        if fft_time < direct_time:
            THRESHOLD = n
            return THRESHOLD

        n = next_size(n)

def find_threshold_dot(trials: int=10000, verbose: bool=True) -> int:
    """ Time both methods to estimate the dot threshold. """
    cdef:
        int n
        double *x
        double *y

    global THRESHOLD_DOT

    n = 2
    while True:
        x = fftw.fftw_alloc_real(n)
        y = fftw.fftw_alloc_real(n)

        direct_time = measure_time(lambda: direct_dot(n, x, y), trials)
        blas_time = measure_time(lambda: blas_dot(n, x, y), trials)

        if verbose:
            print(f"Size {n:4}: direct {direct_time:.3e} blas {blas_time:.3e} "
                  f"ratio {direct_time/blas_time:.3f}")

        # clean up
        fftw.fftw_free(x)
        fftw.fftw_free(y)

        if blas_time < direct_time:
            THRESHOLD_DOT = n
            return THRESHOLD_DOT

        n = next_size(n)

### computational methods

cdef double direct_dot(int n, double *x, double *y):
    """ Inner (dot) product between x and y, <x, y>. """
    cdef:
        int i, incx
        double d

    d = 0
    for i in range(n):
        d += x[i]*y[i]
    return d

cdef double blas_dot(int n, double *x, double *y):
    """ Compute dot product with blas. """
    cdef int incx = 1
    return blas.ddot(&n, x, &incx, y, &incx)

cdef (double (*)(int n, double *x, double *y)) get_dot(int n):
    """ Switch between direct C loop and blas by size. """
    return &direct_dot if n < THRESHOLD_DOT else &blas_dot

cdef void direct_convolve(int n, double *x, int m, double *y):
    """ Compute the convolution between x and y directly. """
    # assume x has m - 1 entries after its entries for a total of n + m - 1
    # the output of this function will overwrite x and leave y untouched
    cdef:
        int i, start, end, shift
        double (*dot)(int n, double *x, double *y)

    dot = get_dot(n)

    # flip y
    for i in range(m//2):
        d = y[i]
        y[i] = y[m - 1 - i]
        y[m - 1 - i] = d

    for i in range(n + m - 2, -1, -1):
        shift = i - (m - 1)
        start = shift if i > m - 1 else 0
        end   = i + 1 if i + 1 < n else n
        x[i] = dot(end - start, x + start, y + start - shift)

cdef void elementwise_product(int n, double *x, const double *y):
    """ Compute the element-wise product between two halfcomplex arrays. """
    # https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
    cdef:
        int half, i
        double x_i

    # i = 0 has no imaginary part
    x[0] *= y[0]
    # skip i = 0, n/2 if n is even
    half = half_ceil(n)
    for i in range(1, half):
        x_i = x[i]
        # update real part
        x[i] = x[i]*y[i] - x[n - i]*y[n - i]
        # update imaginary part
        x[n - i] = x_i*y[n - i] + x[n - i]*y[i]
    # i = n/2 has no imaginary part if n is even
    if n % 2 == 0:
        x[half] *= y[half]

cdef void fft_convolve(int n, double *x, int m, double *y,
                       fftw_plan plan_forward, fftw_plan plan_backward,
                       double *temp):
    """ Compute the convolution between x and y using the fft. """
    cdef:
        int size, i, shift
        double x_0, y_0

    # add the constant interactions x[0]*y + y[0]*x - x[0]*y[0] to temp memory
    # at the same time, remove the constant terms by shifting 1 to the left
    temp[0] = x[0]*y[0]
    x_0 = x[0]
    y_0 = y[0]
    x[0] = x[1]
    y[0] = y[1]
    for i in range(1, n - 1):
        temp[i] = x_0*y[i] + y_0*x[i]
        x[i] = x[i + 1]
        y[i] = y[i + 1]
    temp[n - 1] = x_0*y[n - 1] + y_0*x[n - 1]

    # in-place real-to-real fast Fourier transform
    fftw.fftw_execute_r2r(plan_forward, x, x)
    fftw.fftw_execute_r2r(plan_forward, y, y)
    # in-place element-wise product of x and y, placing the result in x
    elementwise_product(n, x, y)
    # in-place real-to-real inverse fast Fourier transform
    fftw.fftw_execute_r2r(plan_backward, x, x)

    # shift the result 2 to the right and add the result stored in temp
    # scale by 1/n to normalize since the FFT is unnormalizd
    shift = 2
    # this doesn't need to be written into since x is padded by 1
    # but it should be safe since y is after, and y is always at least length 2
    # x[n + 1] = x[n - 1]/n
    x[n] = x[n - 2]/n
    for i in range(n - 1, shift - 1, -1):
        x[i] = x[i - shift]/n + temp[i]
    x[1] = temp[1]
    x[0] = temp[0]

cdef void convolve(int n, double *x, int m, double *y,
                   fftw_plan plan_forward, fftw_plan plan_backward,
                   double *temp):
    """ Switch between direct convolution and fft based on size. """
    # memory is contiguous, [x, n - 2 free entries, y, m - 2 free entries]
    cdef:
        int i, shift, space
        double y_0

    # special case where direct convolution does not have enough padding
    if n == 2:
        y_0 = y[0]
        # this overwrites y[0] since y is adjacent to x
        x[2] = x[1]*y[1]
        x[1] = x[0]*y[1] + y_0*x[1]
        x[0] = x[0]*y_0
    # small enough, use direct
    elif n < THRESHOLD:
        # copy y to be right-aligned in its available space
        shift = m - 2
        for i in range(m + shift - 1, shift - 1, -1):
            y[i] = y[i - shift]
        direct_convolve(n, x, m, y + shift)
    # too large, use fft
    else:
        # clear out empty space
        space = size_space(n)
        for i in range(n, space):
            x[i] = 0
            y[i] = 0
        fft_convolve(space, x, space, y, plan_forward, plan_backward, temp)

cdef void __convolve_pdf(const double[::1] probs, double[::1] pdf):
    """ Compute the probability density function with direct convolve. """
    cdef:
        int n, i, j
        double p, notp

    n = probs.shape[0]
    # 1 probability to have 0 if no draws
    pdf[0] = 1
    for i in range(n):
        p = probs[i]
        notp = 1 - p
        for j in range(i + 1, 0, -1):
            pdf[j] = notp*pdf[j] + p*pdf[j - 1]
        pdf[0] *= notp

def convolve_pdf(probs: np.ndarray) -> np.ndarray:
    """ Compute the probability density function with direct convolve. """
    pdf = np.zeros(probs.shape[0] + 1)
    __convolve_pdf(probs, pdf)
    return pdf

cdef (double *) __pdf(const double[::1] probs, unsigned flags):
    """ Compute the probability density function with fft convolve. """
    cdef:
        int n, i, size, space
        double *temp
        double *polys
        fftw_plan plan_forward, plan_backward

    n = probs.shape[0]
    # pad to nearest power of 2, double
    size = 1 << (<int> ceil(log2(n)) + 1)
    polys = fftw.fftw_alloc_real(size)
    for i in range(n):
        polys[2*i] = 1 - probs[i]
        polys[2*i + 1] = probs[i]
    # malloc doesn't zero out, do manually
    for i in range(2*n, size):
        polys[i] = 0
    # temp storage for fft
    temp = fftw.fftw_alloc_real(size//2)

    size = 2
    while n > 1:
        space = size_space(size)

        # pre-plan for the fft
        if size >= THRESHOLD:
            plan_forward = \
                fftw.fftw_plan_r2r_1d(space, polys, polys,
                                      fftw.FFTW_R2HC, flags)
            plan_backward = \
                fftw.fftw_plan_r2r_1d(space, polys, polys,
                                      fftw.FFTW_HC2R, flags)
        # pair up adjacents, ignoring the last element if not even
        for i in range(0, n - 1, 2):
            convolve(size, polys + i*space, size, polys + (i + 1)*space,
                     plan_forward, plan_backward, temp)
        # if odd, clear out last element's extra space
        if n % 2 != 0:
            for i in range(size, space):
                polys[(n - 1)*space + i] = 0
        # clean up fft
        if size >= THRESHOLD:
            fftw.fftw_destroy_plan(plan_forward)
            fftw.fftw_destroy_plan(plan_backward)

        n = half_ceil(n)
        size = next_size(size)

    # free with fftw free not generic free
    fftw.fftw_free(temp)
    return polys

def pdf(probs: np.ndarray, flags: int=fftw.FFTW_ESTIMATE) -> np.ndarray:
    """ Compute the probability density function with fft convolve. """
    cdef double *polys = __pdf(probs, flags)
    # copy pdf as numpy array and free with fftw free
    pmf = np.copy(<double[:probs.shape[0] + 1]> polys)
    fftw.fftw_free(polys)
    return pmf

