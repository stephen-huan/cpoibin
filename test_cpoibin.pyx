# cython: profile=False
cimport fftw
from fftw cimport fftw_plan
from cpoibin.cpoibin cimport (
    direct_dot,
    blas_dot,
    get_dot,
    direct_convolve,
    elementwise_product,
    fft_convolve,
    convolve,
)
from cpoibin import pdf
import time, typing
import numpy as np
import scipy.signal as signal

# random seed
rng = np.random.default_rng(1)

cdef double[::1] complex_halfcomplex(int n, double complex[::1] x):
    """ Convert a complex array to a halfcomplex array. """
    # complex arrays have adjacent real and imaginary parts for the ith element
    # https://www.fftw.org/fftw3_doc/Complex-numbers.html
    # halfcomplex have the first half real, second half imaginary
    # https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html

    # the size must be provided because the input could have been even or odd
    cdef:
        int i
        double[::1] y

    y = np.zeros(n)
    # i = 0 has no imaginary part
    y[0] = x[0].real
    for i in range(1, n//2 + 1):
        # copy imaginary part
        y[n - i] = x[i].imag
        # copy real part
        y[i] = x[i].real
    # the real part must be copied second so on the last iteration,
    # for a even sized input the real part will be copied properly

    return y

cdef double complex[::1] halfcomplex_complex(double[::1] x):
    """ Convert a halfcomplex array to a complex array. """
    cdef:
        int n, half, i
        double complex[::1] y

    n = x.shape[0]
    half = n//2 + 1
    y = np.zeros(half, dtype=np.complex128)
    # i = 0 has no imaginary part
    y[0].real = x[0]
    y[0].imag = 0
    for i in range(1, half):
        # copy real part
        y[i].real = x[i]
        # copy imaginary part
        y[i].imag = x[n - i]
    # i = n/2 has no imaginary part for even n
    if n % 2 == 0:
        y[half - 1].imag = 0

    return y

def measure_time(function: typing.Callable[[], None], trials: int) -> float:
    """ Time a function over trials iterations. """
    start = time.time()
    for _ in range(trials):
        function()
    return (time.time() - start)/trials

def test_dot(trials: int=200, rows: int=21):
    """ Test dot product against numpy for correctness and speed. """
    cdef:
        int size
        double[::1] x, y
        double (*dot)(int n, double *x, double *y)

    for size in [2**k + 1 for k in range(rows)]:
        x = rng.random(size)
        y = rng.random(size)
        x_numpy = np.asarray(x)
        y_numpy = np.asarray(y)

        numpy_ans = np.dot(x_numpy, y_numpy)

        dot = get_dot(size)
        c_ans = dot(size, &x[0], &y[0])

        assert np.allclose(numpy_ans, c_ans), "methods disagree"

        numpy_time = measure_time(lambda: np.dot(x_numpy, y_numpy), trials)
        c_time = measure_time(lambda: dot(size, &x[0], &y[0]), trials)

        print(f"Size {size:7}: numpy {numpy_time:.3e} c {c_time:.3e} "
              f"ratio {numpy_time/c_time:.3f}")

def test_direct_convolve(trials: int=100, rows: int=13):
    """ Test direct convolution against numpy for correctness and speed. """
    cdef:
        int size
        double[::1] x, y

    for size in [2**k + 1 for k in range(1, rows)]:
        # x is padded to the final size for in-place convolution
        x = rng.random(2*size - 1)
        y = rng.random(size)
        x_numpy = np.asarray(x)
        y_numpy = np.asarray(y)
        # input to numpy needs to not have the extra padding
        x_input = x_numpy[:size]

        numpy_ans = np.convolve(x_input, y_numpy)
        # perform operation in-place, x contains the output
        direct_convolve(size, &x[0], size, &y[0])

        assert np.allclose(numpy_ans, x_numpy), "methods disagree"

        numpy_time = measure_time(
            lambda: np.convolve(x_input, y_numpy), trials
        )
        c_time = measure_time(
            lambda: direct_convolve(size, &x[0], size, &y[0]), trials
        )

        print(f"Size {size:7}: numpy {numpy_time:.3e} c {c_time:.3e} "
              f"ratio {numpy_time/c_time:.3f}")

def test_elementwise(trials: int=200, rows: int=21):
    """ Test elementwise product against numpy for correctness and speed. """
    cdef:
        int size
        double[::1] x, y

    for size in [2**k for k in range(rows)]:
        x = rng.random(size)
        y = rng.random(size)
        x_numpy = np.asarray(x)
        y_numpy = np.asarray(y)
        # interpret both arrays as halfcomplex, give numpy complex
        x_input = np.asarray(halfcomplex_complex(x))
        y_input = np.asarray(halfcomplex_complex(y))

        numpy_ans = complex_halfcomplex(size, x_input*y_input)
        # perform operation in-place, x contains the output
        elementwise_product(size, &x[0], &y[0])

        assert np.allclose(numpy_ans, x_numpy), "methods disagree"

        numpy_time = measure_time(lambda: x_input*y_input, trials)
        c_time = measure_time(
            lambda: elementwise_product(size, &x[0], &y[0]), trials
        )

        print(f"Size {size:7}: numpy {numpy_time:.3e} c {c_time:.3e} "
              f"ratio {numpy_time/c_time:.3f}")

def test_fft(forward: bool=True, trials: int=100, rows: int=21,
             flags: int=fftw.FFTW_ESTIMATE):
    """ Test fftw against numpy for correctness and speed. """
    cdef:
        int size
        double *x
        fftw_plan plan

    for size in [2**k for k in range(1, rows)]:
        x = fftw.fftw_alloc_real(size)
        x_numpy = np.asarray(<double[:size]> x)

        # plan first as plan can overwrite data in input array
        direction = fftw.FFTW_R2HC if forward else fftw.FFTW_HC2R
        plan = fftw.fftw_plan_r2r_1d(size, x, x, direction, flags)

        # fill with reasonable data
        x_numpy[:] = rng.random(size)
        # normalize entries first because fftw is unnormalized
        # l2 norm still grows because natural sqrt(n) from increasing size
        x_numpy /= np.sqrt(size)

        # interpreting x as halfcomplex, also form complex
        x_complex = np.asarray(halfcomplex_complex(x_numpy))

        # force unnormalized fft
        numpy_fft = (lambda x: np.fft.rfft(x, norm="backward")) if forward \
               else (lambda x: np.fft.irfft(x, norm="forward"))

        # for forward, both take the same input and output different formats
        # for backward, take different formats and output the same thing
        numpy_input = x_numpy if forward else x_complex
        numpy_ans = numpy_fft(numpy_input)
        # perform operation in-place, x contains the output
        fftw.fftw_execute_r2r(plan, x, x)

        if forward:
            # numpy outputs as complex and fftw outputs halfcomplex
            numpy_hc = np.asarray(complex_halfcomplex(size, numpy_ans))
            fftw_c = np.asarray(halfcomplex_complex(x_numpy))

            assert np.allclose(numpy_hc, x_numpy), "halfcomplex don't agree"
            assert np.allclose(numpy_ans, fftw_c), "complex don't agree"
        else:
            # both output real values, compare directly
            assert np.allclose(numpy_ans, x_numpy), "real output don't agree"

        numpy_time = measure_time(lambda: numpy_fft(numpy_input), trials)
        fftw_time = measure_time(
            lambda: fftw.fftw_execute_r2r(plan, x, x), trials
        )

        print(f"Size {size:7}: numpy {numpy_time:.3e} fftw {fftw_time:.3e} "
              f"ratio {numpy_time/fftw_time:.3f}")

        # clean up
        fftw.fftw_destroy_plan(plan)
        fftw.fftw_free(x)

def test_fft_convolve(trials: int=10, rows: int=21,
                      flags: int=fftw.FFTW_ESTIMATE):
    """ Test fft convolve against scipy for correctness and speed. """
    cdef:
        int n, size
        double *x
        double *y
        double *temp
        fftw_plan plan_forward, plan_backward

    for n in [2**k for k in range(rows)]:
        # fft needs to be padded to final size, add one
        size = 2*n
        # allocate both inputs at once
        x = fftw.fftw_alloc_real(2*size)
        y = x + size
        x_numpy = np.asarray(<double[:size]> x)
        y_numpy = np.asarray(<double[:size]> y)
        # scipy only needs the unpadded input
        x_input = x_numpy[:n]
        y_input = y_numpy[:n]

        # temporary memory for fft convolve
        temp = fftw.fftw_alloc_real(size)

        # plan first as plan can overwrite data in input array
        plan_forward = \
            fftw.fftw_plan_r2r_1d(size, x, x, fftw.FFTW_R2HC, flags)
        plan_backward = \
            fftw.fftw_plan_r2r_1d(size, x, x, fftw.FFTW_HC2R, flags)

        # fill with reasonable data
        x_numpy[:n] = rng.random(n)
        y_numpy[:n] = rng.random(n)
        # clear padding
        x_numpy[n:] = 0
        y_numpy[n:] = 0

        scipy_ans = signal.fftconvolve(x_input, y_input)
        # perform operation in-place, x contains the output
        fft_convolve(size, x, size, y, plan_forward, plan_backward, temp)

        assert np.allclose(scipy_ans, x_numpy[:size - 1]), "methods disagree"

        scipy_time = measure_time(
            lambda: signal.fftconvolve(x_input, y_input), trials
        )
        fft_time = measure_time(
            lambda: fft_convolve(size, x, size, y,
                                 plan_forward, plan_backward, temp),
            trials
        )

        print(f"Size {n:7}: scipy {scipy_time:.3e} fft {fft_time:.3e} "
              f"ratio {scipy_time/fft_time:.3f}")

        # clean up
        fftw.fftw_destroy_plan(plan_forward)
        fftw.fftw_destroy_plan(plan_backward)
        fftw.fftw_free(x)
        fftw.fftw_free(temp)


def test_convolve(trials: int=10, rows: int=21,
                  flags: int=fftw.FFTW_ESTIMATE):
    """ Test convolve against scipy for correctness and speed. """
    cdef:
        int n, size
        double *x
        double *y
        double *temp
        fftw_plan plan_forward, plan_backward

    for n in [2**k + 1 for k in range(rows)]:
        # fft needs to be padded to final size, add one
        size = 2*(n - 1)
        # allocate both inputs at once
        x = fftw.fftw_alloc_real(2*size)
        y = x + size
        x_numpy = np.asarray(<double[:size]> x)
        y_numpy = np.asarray(<double[:size]> y)
        output_numpy = np.asarray(<double[:2*n - 1]> x)
        # scipy only needs the unpadded input
        x_input = x_numpy[:n]
        y_input = y_numpy[:n]

        # temporary memory for fft convolve
        temp = fftw.fftw_alloc_real(size)

        # plan first as plan can overwrite data in input array
        plan_forward = \
            fftw.fftw_plan_r2r_1d(size, x, x, fftw.FFTW_R2HC, flags)
        plan_backward = \
            fftw.fftw_plan_r2r_1d(size, x, x, fftw.FFTW_HC2R, flags)

        # fill with reasonable data
        x_numpy[:n] = rng.random(n)
        y_numpy[:n] = rng.random(n)
        # clear padding
        x_numpy[n:] = 0
        y_numpy[n:] = 0

        # pick most efficient method for this size
        method = signal.choose_conv_method(x_input, y_input)
        scipy_ans = signal.convolve(x_input, y_input, method=method)
        # perform operation in-place, x contains the output
        convolve(n, x, n, y, plan_forward, plan_backward, temp)

        assert np.allclose(scipy_ans, output_numpy), "methods disagree"

        scipy_time = measure_time(
            lambda: signal.convolve(x_input, y_input, method=method), trials
        )
        c_time = measure_time(
            lambda: convolve(n, x, n, y, plan_forward, plan_backward, temp),
            trials
        )

        print(f"Size {n:7}: scipy {scipy_time:.3e} c {c_time:.3e} "
              f"ratio {scipy_time/c_time:.3f}")

        # clean up
        fftw.fftw_destroy_plan(plan_forward)
        fftw.fftw_destroy_plan(plan_backward)
        fftw.fftw_free(x)
        fftw.fftw_free(temp)

def python_pdf(probs: np.ndarray) -> np.ndarray:
    """ Compute the probability density function with scipy convolve. """
    polys = [np.array([1 - p, p]) for p in probs]
    while len(polys) > 1:
        # pick most efficient method for this size
        method = signal.choose_conv_method(polys[0], polys[1])
        # pair up adjacents, copying the last element if not even
        polys = [signal.convolve(polys[i], polys[i + 1], method=method)
                for i in range(0, len(polys) - 1, 2)] + \
            ([polys[len(polys) - 1]] if len(polys) % 2 != 0 else [])
    return polys[0]

def test_pdf(trials: int=10, rows: int=17):
    """ Test final pdf against scipy for correctness and speed. """
    for size in [2**k for k in range(rows)]:
        probs = rng.random(size)

        python_ans = python_pdf(probs)
        c_ans = pdf(probs)

        assert np.allclose(python_ans, c_ans), "methods disagree"

        python_time = measure_time(lambda: python_pdf(probs), trials)
        c_time = measure_time(lambda: pdf(probs), trials)

        print(f"Size {size:7}: python {python_time:.3e} c {c_time:.3e} "
              f"ratio {python_time/c_time:.3f}")

