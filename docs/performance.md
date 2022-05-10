# Performance Comparison

Comparisons are against the original
[repository](https://github.com/tsakim/poibin), as well as against standard
scientific Python functions (`numpy.dot`, `numpy.fft.rfft`, `numpy.convolve`,
`scipy.signal.fftconvolve` `scipy.signal.convolve`, etc.).

## Asymptotic Time and Space Complexities

The algorithm implemented in this repository has a time complexity of O(n log^2
n) to compute the probability mass function of the sum of `n` independent (but
not necessarily identically distributed) Bernoulli random variables (number
of probabilities given). The time complexity comes from the fact that the
algorithm must perform n/2 convolutions of length 2, n/4 of length 4, n/8 of
length 8, and so on, until it computes 2 convolutions of length n/2 for the
final answer. The time complexity to compute a convolution of length `k` is
O(k log k), so the cost on the kth "level" of the algorithm is performing n/k
convolution of length k, (n/k)*O(k log k) is O(n log k). Finally, summing over
k = 2 to k = n, a total of log 2 levels, results in n(log 2 + log 4 + log 8 +
... + log n), which is n(1 + 2 + 3 + ... + log n), which is O(n log^2 n).

The space complexity must be at least O(n), since linear memory is required to
write down the answer. If the input is written as `n` success probabilities
`[p_1, p_2, ..., p_n]`, then the algorithm first expands each probability into
`[1 - p_1, p_1, 1 - p_2, p_2, ..., 1 - p_n, p_n]`, which consumes double the
space of the original input. It then pads to the nearest larger power of 2,
which at most doubles the size of the array again. Finally, it needs extra
temporary memory for convolution by fast Fourier transform, which needs to be
half the size of the padded array. The most the memory can be is `4n` from
doubling and padding to the nearest larger power of 2, and `1/2 (4n)` for
temporary space is at most `6n` additional memory in addition to the input.

The algorithm implemented in the original repository has time complexity
O(n^2) since it must evaluate the degree n polynomial defined as
the product of each pmf of each random variable at n + 1 points:
```text
P(x) = ((1 - p_1) + p_1 x) * ((1 - p_2) + p_2 x) * ... * ((1 - p_n) + p_n x)
```
The ith term of the polynomial `P(x)` contains the probability of the sum being
i as its coefficient. That is, looking at the term `c_i x^i` in the polynomial
`P(x)`, `c_i` is the desired probability for i successes. We can solve for the
coefficients of the polynomial by plugging in n + 1 distinct points `x_0, x_1,
..., x_n` into `P(x)` and solving the corresponding linear system:
```text
|1 x_0 x_0^2 ... x_1^n| |c_0|   |y_0|
|1 x_1 x_1^2 ... x_0^n| |c_1|   |y_1|
|.  .    .         .  | | . |   | . |
|.  .    .         .  | | . | = | . |
|.  .    .         .  | | . |   | . |
|1 x_n x_n^2 ... x_n^n| |c_n|   |y_n|
```
where `x_0, ..., x_n` are the points, `y_0, ..., y_n` are the associated
outputs (i.e. `y_i = P(x_i)`), and `c_0, c_1, ..., c_n` are the coefficients
of `P(x)` (i.e. `c_i x^i` is the ith term in `P(x)`). Thus, from n + 1
evaluations, it is possible to recover the coefficients of the polynomial.
It is typical to choose the complex roots of unity for the points, in which
case the coefficients of the polynomial can be recovered by an inverse Fourier
transform of the `y` vector in time O(n log n). Similarly, a polynomial whose
coefficients are known can be evaluated at n roots of unity with the Fourier
transform of the `c` vector, in time O(n log n). This seems to suggest an
overall O(n log n) algorithm where the polynomial is evaluated at the roots
of unity with the Fourier transform, and then its coefficients recovered
with the inverse Fourier transform. However, the entire problem is that we
are not given the coefficients of the polynomial, and instead we only have
its representation as the product of n binomial. It therefore takes O(n) to
evaluate the polynomial by plugging in the input point into each binomial
and multiplying all together, for a total of O(n^2) for n points, even if
the points are roots of unity.

We can see where this O(n^2) cost of evaluating the polynomial happens in the
[code](https://github.com/tsakim/poibin/blob/51a8efac53e997bed91fd110363ad527f2fbe054/poibin.py#L217-L220):
```python
        # get_z:
        exp_value = np.exp(self.omega * idx_array * 1j)
        xy = 1 - self.success_probabilities + \
            self.success_probabilities * exp_value[:, np.newaxis]
```
`self.success_probabilities` is a length n vector, and so is
`exp_value`. When a new axis is added, it becomes shape (n,
1), and when added to a length n vector the resulting array is
[broadcast](https://numpy.org/doc/stable/user/basics.broadcasting.html) to
be (n, n), a n x n matrix. The way this code is written also uses O(n^2)
space to write down the entries of the matrix, but it possible to perform
the matrix-vector product with no additional space other than the output
by computing entries of the matrix when needed. Of course, in Python
this would lead to a speed penalty because vectorization is lost.

## Empirical Ablation Study

At a high level, the main work of the code is computing convolutions. There
are two main approaches, a direct "sliding dot-product" approach (O(n^2))
and a fast Fourier transform-based approach (O(n log n)). It's faster to use
the simpler method for smaller lists. The following sections compare each
of the intermediate operations in computing convolutions and finishes with
a comparison of the final probability mass function computation.

### Dot Product

The dot product between two vectors is the sum of their
element-wise product. It can be written in Cython as:
```cython
cdef double direct_dot(int n, double *x, double *y):
    """ Inner (dot) product between x and y, <x, y>. """
    cdef:
        int i, incx
        double d

    d = 0
    for i in range(n):
        d += x[i]*y[i]
    return d
```

However, for large vectors, using the [BLAS](https://www.netlib.org/blas/)
library is more efficient. This is likely because of parallelization.

```cython
cdef double blas_dot(int n, double *x, double *y):
    """ Compute dot product with blas. """
    cdef int incx = 1
    return blas.ddot(&n, x, &incx, y, &incx)
```

The code adaptively picks the right method to
use, switching to BLAS once the size reaches 33:
```cython
cdef (double (*)(int n, double *x, double *y)) get_dot(int n):
    """ Switch between direct C loop and blas by size. """
    return &direct_dot if n < THRESHOLD_DOT else &blas_dot
```

| size | direct    | blas      | ratio |
|-----:|:---------:|:---------:|:-----:|
| 2    | 4.270e-08 | 7.482e-08 | 0.571 |
| 3    | 4.292e-08 | 6.368e-08 | 0.674 |
| 5    | 4.270e-08 | 6.580e-08 | 0.649 |
| 9    | 1.042e-07 | 1.157e-07 | 0.900 |
| 17   | 1.525e-07 | 1.815e-07 | 0.840 |
| 33   | 2.341e-07 | 1.739e-07 | 1.346 |
| 65   | 2.205e-07 | 1.215e-07 | 1.815 |
| 129  | 8.695e-07 | 6.811e-07 | 1.277 |
| 257  | 3.871e-07 | 8.438e-08 | 4.588 |
| 513  | 2.098e-06 | 1.768e-06 | 1.187 |
| 1025 | 1.416e-05 | 5.443e-06 | 2.601 |
| 2049 | 2.843e-06 | 4.116e-07 | 6.907 |
| 4097 | 5.614e-06 | 1.312e-06 | 4.277 |
| 8193 | 7.635e-05 | 1.540e-05 | 4.958 |

For a comparison against `numpy.dot`, it's much faster for
smaller lists, and asymptotes towards 1 for larger lists.

| size    | numpy     | cython    | ratio  |
|--------:|:---------:|:---------:|:------:|
| 2       | 1.915e-06 | 5.126e-08 | 37.349 |
| 3       | 1.869e-06 | 5.007e-08 | 37.333 |
| 5       | 1.891e-06 | 4.888e-08 | 38.683 |
| 9       | 1.888e-06 | 5.007e-08 | 37.714 |
| 17      | 1.944e-06 | 6.557e-08 | 29.655 |
| 33      | 1.900e-06 | 9.060e-08 | 20.974 |
| 65      | 1.945e-06 | 3.755e-07 | 5.181  |
| 129     | 1.879e-06 | 9.894e-08 | 18.988 |
| 257     | 1.884e-06 | 1.156e-07 | 16.289 |
| 513     | 1.991e-06 | 1.395e-07 | 14.274 |
| 1025    | 2.170e-06 | 2.158e-07 | 10.055 |
| 2049    | 2.741e-06 | 5.758e-07 | 4.760  |
| 4097    | 4.705e-06 | 1.760e-06 | 2.674  |
| 8193    | 4.860e-06 | 2.334e-06 | 2.082  |
| 16385   | 7.905e-06 | 4.436e-06 | 1.782  |
| 32769   | 9.905e-06 | 4.755e-06 | 2.083  |
| 65537   | 1.408e-05 | 9.176e-06 | 1.534  |
| 131073  | 2.983e-05 | 2.542e-05 | 1.174  |

### Direct Convolution

The bottleneck in brute-force convolution computation is the dot
product. Similar to the dot product, against `numpy.convolve` it
is much faster for smaller lists and the same for larger lists.

| size | numpy     | cython    | ratio  |
|-----:|:---------:|:---------:|:------:|
| 3    | 6.599e-06 | 1.407e-07 | 46.915 |
| 5    | 6.242e-06 | 2.003e-07 | 31.167 |
| 9    | 6.931e-06 | 3.099e-07 | 22.362 |
| 17   | 7.329e-06 | 6.628e-07 | 11.058 |
| 33   | 9.227e-06 | 2.999e-06 | 3.076  |
| 65   | 1.251e-05 | 6.001e-06 | 2.085  |
| 129  | 2.050e-05 | 1.245e-05 | 1.646  |
| 257  | 4.192e-05 | 3.054e-05 | 1.373  |
| 513  | 9.858e-05 | 8.256e-05 | 1.194  |
| 1025 | 3.031e-04 | 2.702e-04 | 1.122  |
| 2049 | 1.035e-03 | 9.023e-04 | 1.147  |
| 4097 | 4.757e-03 | 4.686e-03 | 1.015  |

### Fast Fourier Transform (FFT)

The fast Fourier transform is computed with the library
[fftw](https://www.fftw.org/). There are two variables that
significantly affect performance. One is the mechanism to store
pre-planning for particular sizes. This mechanism is called
[wisdom](https://www.fftw.org/fftw3_doc/Wisdom.html#Wisdom) and
involves a few hours of pre-computing the optimal algorithms
("plans") to compute a transform for a particular size. The second
variable is parallelization, which is provided through fftw's
[OpenMP](https://www.openmp.org/) support. The experiments are run on
a 8 core Intel i9-9880H @ 2.30GHz CPU with 16 virtual cores.

Parallelism can be disabled in numpy following [this Stack Overflow answer](
https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy).
It suffices to set the following environmental variables:
```shell
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

no wisdom, no openmp

| size    | numpy     | fftw      | ratio   |
|--------:|:---------:|:---------:|:-------:|
| 2       | 3.178e-06 | 4.768e-08 | 66.650  |
| 4       | 2.980e-06 | 3.099e-08 | 96.154  |
| 8       | 3.080e-06 | 2.861e-08 | 107.667 |
| 16      | 3.200e-06 | 5.245e-08 | 61.000  |
| 32      | 3.300e-06 | 6.914e-08 | 47.724  |
| 64      | 3.531e-06 | 2.193e-07 | 16.098  |
| 128     | 4.079e-06 | 5.722e-07 | 7.129   |
| 256     | 4.761e-06 | 8.702e-07 | 5.471   |
| 512     | 6.468e-06 | 1.922e-06 | 3.366   |
| 1024    | 9.720e-06 | 4.230e-06 | 2.298   |
| 2048    | 1.817e-05 | 9.398e-06 | 1.933   |
| 4096    | 3.670e-05 | 2.152e-05 | 1.705   |
| 8192    | 7.240e-05 | 5.898e-05 | 1.228   |
| 16384   | 1.565e-04 | 1.364e-04 | 1.148   |
| 32768   | 3.909e-04 | 3.002e-04 | 1.302   |
| 65536   | 8.597e-04 | 9.540e-04 | 0.901   |
| 131072  | 1.777e-03 | 2.019e-03 | 0.880   |
| 262144  | 4.253e-03 | 4.053e-03 | 1.049   |
| 524288  | 9.566e-03 | 1.553e-02 | 0.616   |
| 1048576 | 1.872e-02 | 3.282e-02 | 0.570   |
| 2097152 | 4.646e-02 | 9.658e-02 | 0.481   |

Without wisdom, the `FFTW_ESTIMATE` flag means to use heuristics to
get a decent algorithm. It is much faster than `numpy.fft.rfft` for
small lists, but for lists larger than 500,000 it actually becomes
twice as slow as numpy. This issue can be alleviated with wisdom.

exhaustive wisdom, no openmp

| size    | numpy     | fftw      | ratio  |
|--------:|:---------:|:---------:|:------:|
| 2       | 2.990e-06 | 5.245e-08 | 57.000 |
| 4       | 2.880e-06 | 3.099e-08 | 92.923 |
| 8       | 2.849e-06 | 3.099e-08 | 91.923 |
| 16      | 3.040e-06 | 4.053e-08 | 75.000 |
| 32      | 3.090e-06 | 6.914e-08 | 44.690 |
| 64      | 3.343e-06 | 1.192e-07 | 28.040 |
| 128     | 3.920e-06 | 2.885e-07 | 13.587 |
| 256     | 4.690e-06 | 7.296e-07 | 6.428  |
| 512     | 6.280e-06 | 1.590e-06 | 3.949  |
| 1024    | 9.811e-06 | 3.448e-06 | 2.846  |
| 2048    | 1.644e-05 | 7.360e-06 | 2.234  |
| 4096    | 3.638e-05 | 1.682e-05 | 2.163  |
| 8192    | 7.686e-05 | 3.925e-05 | 1.958  |
| 16384   | 1.526e-04 | 8.763e-05 | 1.741  |
| 32768   | 3.933e-04 | 2.021e-04 | 1.946  |
| 65536   | 8.605e-04 | 4.611e-04 | 1.866  |
| 131072  | 1.986e-03 | 1.026e-03 | 1.936  |
| 262144  | 4.526e-03 | 2.271e-03 | 1.993  |
| 524288  | 9.763e-03 | 4.656e-03 | 2.097  |
| 1048576 | 1.863e-02 | 1.211e-02 | 1.539  |
| 2097152 | 4.917e-02 | 2.837e-02 | 1.733  |

"Exhaustive" wisdom means the `FFTW_EXHAUSTIVE` flag is used in planning.
This allows fftw to be consistently about twice as fast as numpy. However,
the sizes must be pre-planned. Using parallelism can guarantee performance
improvements without heavy preprocessing.

In the following parallel experiments, 8 openmp threads are used.

no wisdom, openmp with 8 threads

| size    | numpy     | fftw      | ratio  |
|--------:|:---------:|:---------:|:------:|
| 2       | 2.892e-06 | 5.245e-08 | 55.136 |
| 4       | 2.880e-06 | 3.099e-08 | 92.923 |
| 8       | 2.930e-06 | 3.099e-08 | 94.538 |
| 16      | 3.071e-06 | 4.053e-08 | 75.765 |
| 32      | 3.152e-06 | 6.199e-08 | 50.846 |
| 64      | 3.569e-06 | 5.676e-05 | 0.063  |
| 128     | 4.098e-06 | 1.584e-04 | 0.026  |
| 256     | 4.859e-06 | 5.320e-05 | 0.091  |
| 512     | 6.509e-06 | 5.446e-05 | 0.120  |
| 1024    | 9.699e-06 | 5.623e-05 | 0.172  |
| 2048    | 1.665e-05 | 6.028e-05 | 0.276  |
| 4096    | 3.295e-05 | 7.451e-05 | 0.442  |
| 8192    | 7.240e-05 | 1.754e-04 | 0.413  |
| 16384   | 1.931e-04 | 1.901e-04 | 1.016  |
| 32768   | 3.531e-04 | 2.126e-04 | 1.661  |
| 65536   | 8.439e-04 | 4.668e-04 | 1.808  |
| 131072  | 1.765e-03 | 6.646e-04 | 2.656  |
| 262144  | 4.458e-03 | 1.136e-03 | 3.926  |
| 524288  | 9.266e-03 | 8.331e-03 | 1.112  |
| 1048576 | 1.890e-02 | 1.378e-02 | 1.372  |
| 2097152 | 4.682e-02 | 2.543e-02 | 1.841  |

Performance suffers heavily for lists between sizes 64 and 8192
because of the significant overhead in thread synchronization.
This issue can be overcome with wisdom, as planning with flags
above `FFTW_PATIENT` allows wisdom to [automatically disable
threads for sizes that don't benefit from parallelization](
https://www.fftw.org/fftw3_doc/How-Many-Threads-to-Use_003f.html).

exhaustive wisdom (up to 4096), openmp with 8 threads

| size    | numpy     | fftw      | ratio  |
|--------:|:---------:|:---------:|:------:|
| 2       | 5.288e-06 | 7.868e-08 | 67.212 |
| 4       | 5.059e-06 | 5.245e-08 | 96.455 |
| 8       | 5.231e-06 | 5.960e-08 | 87.760 |
| 16      | 5.438e-06 | 7.868e-08 | 69.121 |
| 32      | 5.760e-06 | 1.287e-07 | 44.741 |
| 64      | 6.151e-06 | 2.313e-07 | 26.598 |
| 128     | 7.119e-06 | 5.722e-07 | 12.442 |
| 256     | 8.519e-06 | 1.547e-06 | 5.505  |
| 512     | 1.174e-05 | 4.342e-06 | 2.705  |
| 1024    | 1.823e-05 | 9.201e-06 | 1.981  |
| 2048    | 3.225e-05 | 1.535e-05 | 2.101  |
| 4096    | 5.981e-05 | 2.456e-05 | 2.435  |
| 8192    | 1.292e-04 | 6.513e-05 | 1.983  |
| 16384   | 2.752e-04 | 8.090e-05 | 3.402  |
| 32768   | 6.710e-04 | 1.099e-04 | 6.108  |
| 65536   | 1.522e-03 | 5.939e-04 | 2.563  |
| 131072  | 3.472e-03 | 9.777e-04 | 3.552  |
| 262144  | 8.376e-03 | 1.871e-03 | 4.476  |
| 524288  | 1.759e-02 | 1.041e-02 | 1.690  |
| 1048576 | 3.297e-02 | 1.938e-02 | 1.702  |
| 2097152 | 7.660e-02 | 3.393e-02 | 2.257  |

In this case, wisdom is only generated for lists up to size 4096.
`FFTW_ESTIMATE` is used for all lists above that size, but the performance
is still as good or often better than both numpy and the single-thread
case. This means that parallelism allows fftw to "generalize", that is,
to achieve good performance without preprocessing.

We neglect to discuss in depth the inverse FFT since it is similar to the
forward transform. The difference is that even in the worst case (no wisdom, no
parallelism) fftw is faster than `np.fft.irfft`. Of course, using exhaustive
wisdom and using multiple threads improves performance further. The detailed
results can be found [here](./footnotes.md#backward-fft).

### Complex Element-wise Product

After the FFT is applied to both inputs, they must be multiplied element-wise.
This is a bit tricky since the arrays are not stored as standard
[complex arrays](https://www.fftw.org/fftw3_doc/Complex-numbers.html),
but in the [halfcomplex](
https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html)
format for efficiency.

| Size | numpy     | cython    | ratio  |
|-----:|:---------:|:---------:|:------:|
| 1    | 7.045e-07 | 3.576e-08 | 19.700 |
| 2    | 7.010e-07 | 3.457e-08 | 20.276 |
| 4    | 7.188e-07 | 4.053e-08 | 17.735 |
| 8    | 7.105e-07 | 3.934e-08 | 18.061 |
| 16   | 7.200e-07 | 5.007e-08 | 14.381 |
| 32   | 7.391e-07 | 6.914e-08 | 10.690 |
| 64   | 7.701e-07 | 1.013e-07 | 7.600  |
| 128  | 9.465e-07 | 1.752e-07 | 5.401  |
| 256  | 1.115e-06 | 3.362e-07 | 3.316  |
| 512  | 1.340e-06 | 6.402e-07 | 2.093  |
| 1024 | 1.960e-06 | 1.204e-06 | 1.628  |
| 2048 | 3.049e-06 | 2.395e-06 | 1.273  |
| 4096 | 5.205e-06 | 4.795e-06 | 1.086  |

As is typical for these linear-time operations, Cython is much faster
than numpy for small lists (as it avoids the Python interpreter overhead)
but for large lists, it asymptotes to the same performance as numpy.

### Convolution by FFT

| size    | scipy     | fftw      | ratio   | fftw (omp) | ratio   |
|--------:|:---------:|:---------:|:-------:|:----------:|:-------:|
| 1       | 7.510e-06 | 2.861e-07 | 25.833  | 3.099e-07  | 24.231  |
| 2       | 7.169e-05 | 1.907e-07 | 377.625 | 1.907e-07  | 375.875 |
| 4       | 7.021e-05 | 1.907e-07 | 371.625 | 2.146e-07  | 327.222 |
| 8       | 7.119e-05 | 1.907e-07 | 373.375 | 3.099e-07  | 229.692 |
| 16      | 7.122e-05 | 4.053e-07 | 176.118 | 3.815e-07  | 186.687 |
| 32      | 7.141e-05 | 6.914e-07 | 103.552 | 7.153e-07  | 99.833  |
| 64      | 7.412e-05 | 1.502e-06 | 49.730  | 1.597e-06  | 46.403  |
| 128     | 7.560e-05 | 3.290e-06 | 23.435  | 3.600e-06  | 21.000  |
| 256     | 8.121e-05 | 6.413e-06 | 14.145  | 8.297e-06  | 9.787   |
| 512     | 8.991e-05 | 1.349e-05 | 6.928   | 1.700e-05  | 5.289   |
| 1024    | 1.081e-04 | 3.011e-05 | 3.862   | 3.369e-05  | 3.209   |
| 2048    | 1.509e-04 | 6.320e-05 | 2.473   | 6.192e-05  | 2.437   |
| 4096    | 2.519e-04 | 1.400e-04 | 1.960   | 1.458e-04  | 1.727   |
| 8192    | 4.641e-04 | 3.538e-04 | 1.455   | 2.434e-04  | 1.907   |
| 16384   | 1.306e-03 | 8.766e-04 | 1.127   | 3.944e-04  | 3.312   |
| 32768   | 2.872e-03 | 1.721e-03 | 1.813   | 1.402e-03  | 2.049   |
| 65536   | 7.325e-03 | 3.778e-03 | 2.090   | 2.134e-03  | 3.432   |
| 131072  | 1.368e-02 | 9.140e-03 | 1.845   | 4.454e-03  | 3.070   |
| 262144  | 2.966e-02 | 1.906e-02 | 1.665   | 1.566e-02  | 1.894   |
| 524288  | 5.656e-02 | 4.336e-02 | 1.325   | 3.475e-02  | 1.627   |
| 1048576 | 1.805e-01 | 9.244e-02 | 1.637   | 1.036e-01  | 1.743   |
| 2097152 | 4.369e-01 | 1.960e-01 | 1.641   | 1.986e-01  | 2.200   |

Comparing against `scipy.signal.fftconvolve`, the entire convolution
operation (transform of both, element-wise product, then inverse
transform) essentially inherits the speed of fftw compared to numpy.

### Convolution

| size    | scipy     | cython    | ratio  | cython (omp) | ratio  |
|--------:|:---------:|:---------:|:------:|:------------:|:------:|
| 2       | 8.512e-06 | 2.146e-07 | 20.444 | 5.245e-07    | 16.227 |
| 3       | 6.795e-06 | 1.192e-07 | 31.200 | 1.907e-07    | 35.625 |
| 5       | 6.890e-06 | 9.537e-08 | 41.000 | 2.146e-07    | 32.111 |
| 9       | 7.319e-06 | 1.907e-07 | 21.125 | 3.099e-07    | 23.615 |
| 17      | 8.225e-06 | 3.099e-07 | 14.846 | 7.153e-07    | 11.500 |
| 33      | 9.799e-06 | 7.153e-07 | 7.667  | 1.287e-06    | 7.611  |
| 65      | 1.349e-05 | 1.407e-06 | 5.203  | 2.789e-06    | 4.838  |
| 129     | 2.182e-05 | 3.314e-06 | 3.612  | 6.557e-06    | 3.327  |
| 257     | 4.551e-05 | 6.413e-06 | 3.788  | 1.857e-05    | 2.451  |
| 513     | 1.074e-04 | 1.400e-05 | 4.174  | 4.020e-05    | 2.671  |
| 1025    | 3.031e-04 | 3.209e-05 | 4.993  | 6.230e-05    | 4.865  |
| 2049    | 9.215e-04 | 7.091e-05 | 7.467  | 1.076e-04    | 8.564  |
| 4097    | 4.907e-04 | 1.538e-04 | 1.803  | 2.382e-04    | 2.060  |
| 8193    | 9.506e-04 | 3.763e-04 | 1.443  | 4.320e-04    | 2.201  |
| 16385   | 1.810e-03 | 7.985e-04 | 1.413  | 6.710e-04    | 2.697  |
| 32769   | 4.343e-03 | 1.754e-03 | 1.531  | 2.367e-03    | 1.835  |
| 65537   | 1.036e-02 | 3.682e-03 | 1.563  | 4.195e-03    | 2.469  |
| 131073  | 2.510e-02 | 9.915e-03 | 1.366  | 8.424e-03    | 2.979  |
| 262145  | 4.715e-02 | 1.883e-02 | 1.631  | 3.164e-02    | 1.490  |
| 524289  | 9.154e-02 | 4.444e-02 | 1.362  | 5.755e-02    | 1.591  |
| 1048577 | 2.087e-01 | 9.371e-02 | 1.640  | 1.081e-01    | 1.931  |
| 2097153 | 4.416e-01 | 2.013e-01 | 1.574  | 2.045e-01    | 2.160  |

Finally, we compare against `scipy.signal.convolve` which switches its
method from direct to FFT if the list is large enough (on my computer,
somewhere between 1025 to 4097). For my Cython implementation of direct
and FFT, it's faster to use the FFT for anything larger than 17.

### PMF Computation

Finally, we do an end-to-end analysis of the entire system, comparing
against a direct Python implementation as well as the original repository.

#### Comparison against Python

First, we compare against the following pure-Python implementation:
```python
import numpy as np
import scipy.signal as signal

def pdf(probs: np.ndarray) -> np.ndarray:
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
```

Note that the time complexity of this code is also O(n log^2 n). Thus, the
only speedups Cython can achieve is better constant-factor performance.

| size    | python    | cython    | ratio  | cython (omp) | ratio  |
|--------:|:---------:|:---------:|:------:|:------------:|:------:|
| 1       | 3.695e-06 | 4.482e-06 | 0.468  | 8.202e-06    | 0.451  |
| 2       | 3.741e-05 | 4.387e-06 | 4.397  | 7.892e-06    | 4.740  |
| 4       | 7.849e-05 | 4.125e-06 | 9.335  | 8.893e-06    | 8.826  |
| 8       | 1.331e-04 | 4.411e-06 | 14.108 | 9.108e-06    | 14.615 |
| 16      | 1.871e-04 | 4.697e-06 | 23.142 | 8.583e-06    | 21.800 |
| 32      | 3.287e-04 | 2.239e-05 | 8.558  | 4.070e-05    | 8.076  |
| 64      | 6.848e-04 | 4.191e-05 | 8.348  | 1.666e-04    | 4.110  |
| 128     | 1.332e-03 | 6.189e-05 | 10.718 | 2.743e-04    | 4.856  |
| 256     | 2.370e-03 | 1.989e-04 | 6.402  | 6.792e-04    | 3.490  |
| 512     | 4.648e-03 | 3.455e-04 | 7.320  | 8.851e-04    | 5.252  |
| 1024    | 9.145e-03 | 5.533e-04 | 8.990  | 1.337e-03    | 6.842  |
| 2048    | 1.910e-02 | 9.031e-04 | 11.643 | 2.614e-03    | 7.309  |
| 4096    | 4.025e-02 | 1.437e-03 | 15.667 | 4.730e-03    | 8.508  |
| 8192    | 8.114e-02 | 2.365e-03 | 19.169 | 7.471e-03    | 10.860 |
| 16384   | 1.635e-01 | 5.077e-03 | 17.520 | 1.189e-02    | 13.756 |
| 32768   | 3.278e-01 | 8.340e-03 | 22.109 | 1.930e-02    | 16.985 |
| 65536   | 6.613e-01 | 1.848e-02 | 20.093 | 3.421e-02    | 19.334 |
| 131072  | 1.335e+00 | 3.813e-02 | 19.735 | 4.611e-02    | 28.962 |
| 262144  | 2.689e+00 | 8.726e-02 | 17.472 | 1.335e-01    | 20.145 |
| 524288  | 5.291e+00 | 1.930e-01 | 16.009 | 1.629e-01    | 32.473 |
| 1048576 | 6.213e+00 | 4.349e-01 | 14.497 | 3.361e-01    | 18.486 |
| 2097152 | 1.649e+01 | 1.048e+00 | 12.323 | 1.303e+00    | 12.658 |

We see significant improvement, around 8x for "small" lists
(8-4096) and 20x for medium sized lists (8192-131072).
However, for very large lists the advantage seems to shrink.

#### Comparison against poibin

We use the methodology in [1]. That is, we pick probabilities
uniformly in [0, 1] and sample 1000 times (we only sample 10 times)
for each size. The different sizes are taken from the reference.

The code can be found in [examples/comparison.py](../examples/comparison.py).

For non-parallel execution, we run:
```shell
OMP_NUM_THREADS=1 python comparison.py
```
For parallel execution, we instead run:
```shell
OMP_NUM_THREADS=8 python comparison.py
```

| size  | poibin    | cython    | ratio   | cython (omp) | ratio   |
|------:|:---------:|:---------:|:-------:|:------------:|:-------:|
| 10    | 1.244e-04 | 1.026e-04 | 1.213   | 1.040e-04    | 1.197   |
| 20    | 1.377e-04 | 1.307e-04 | 1.053   | 1.313e-04    | 1.046   |
| 50    | 2.102e-04 | 1.701e-04 | 1.236   | 2.287e-04    | 0.970   |
| 100   | 4.150e-04 | 2.212e-04 | 1.876   | 3.231e-04    | 1.297   |
| 200   | 1.099e-03 | 4.163e-04 | 2.640   | 5.531e-04    | 2.024   |
| 500   | 5.741e-03 | 7.333e-04 | 7.828   | 8.785e-04    | 6.766   |
| 1000  | 2.466e-02 | 1.244e-03 | 19.828  | 1.502e-03    | 16.818  |
| 2000  | 8.680e-02 | 2.102e-03 | 41.301  | 2.435e-03    | 35.764  |
| 5000  | 5.727e-01 | 4.931e-03 | 116.140 | 7.061e-03    | 82.123  |
| 10000 | 2.829e+00 | 9.739e-03 | 290.448 | 1.378e-02    | 236.593 |
| 15000 | 7.280e+00 | 1.250e-02 | 582.467 | 1.937e-02    | 473.561 |

We see significant speedups, which get larger and larger as the size
increases (as the gap between O(n log^2 n) and O(n^2) becomes larger and
larger). Unfortunately, the sizes are not large enough to see a speedup
from parallelization (which happens around a size of 100,000).

[[1] Yili Hong, On computing the distribution function for the Poisson binomial
distribution, Computational Statistics & Data Analysis, Volume 59, March 2013,
pages 41-51, ISSN 0167-9473](https://doi.org/10.1016/j.csda.2012.10.006)

