from test_cpoibin import (
    test_dot,
    test_direct_convolve,
    test_elementwise,
    test_fft,
    test_fft_convolve,
    test_convolve,
    test_pdf,
)

if __name__ == "__main__":
    print("dot product")
    test_dot(rows=23)

    # cython implementation seems to be slow versus numpy
    # https://github.com/numpy/numpy/blob/06f6cc22d79203cbdb1ad43cc141853e24cb120c/numpy/core/src/multiarray/multiarraymodule.c#L1177-L1299
    # fixed, solution is to use blas for dot product
    print("direct convolution")
    test_direct_convolve(rows=13)

    print("complex elementwise product")
    test_elementwise(rows=23)

    print("forward fft")
    test_fft(forward=True, rows=22)

    print("backward fft")
    test_fft(forward=False, rows=22)

    print("fft convolution")
    test_fft_convolve(rows=22)

    print("convolution")
    test_convolve(rows=22)

    print("pdf")
    test_pdf(rows=22)

