# Poisson binomial (poibin) for [P/C]ython

## About

This is a performance-oriented Cython fork of
[tsakim/poibin](https://github.com/tsakim/poibin).

The module contains a [Cython](https://cython.org/) implementation of
functions related to the Poisson binomial probability distribution [1],
which describes the probability distribution of the sum of independent
Bernoulli random variables with non-uniform success probabilities. For
further information, see reference [1].

The algorithm implemented is O(n log^2 n) where `n` is the number of random
variables. At a high level the approach is from reference [2], but many
specific optimizations are made for the Poisson binomial case. The key idea
is to treat each random variable's probability mass function (pmf) as a
polynomial and multiply (convolve) the polynomials to get the pmf of the sum
of the independent random variables. Recursive application of this procedure
yields the final algorithm. Finally, the fast Fourier transform (fft) is
used to accelerate each convolution. For a detailed performance comparison
and ablation study, see [performance.md](./docs/performance.md).

The implemented methods are:
- `pmf`: probability mass function
- `cdf`: cumulative distribution function
- `pval`: p-value for right tailed tests

## Dependencies

Dependencies are managed with [conda](https://conda.io/).
Install dependencies from `environment.yml` with:
```shell
conda env create --prefix ./venv --file environment.yml
```
Enter the created virtual environment with:
```shell
conda activate ./venv
```
and finally, compile the Cython extensions:
```shell
python setup.py build_ext --inplace
```

The dependencies are:
- [python](https://www.python.org/)
- [cython](https://cython.org/) (build dependency)
- [pygments](https://pygments.org/) (syntax highlighting in Cython html output)
- [pytest](https://docs.pytest.org/en/latest/contents.html) (for testing)
- [numpy](http://www.numpy.org/)
- [scipy](https://scipy.org/)
- [fftw](https://www.fftw.org/) (fast FFT computation)
- [llvm-openmp](https://www.openmp.org/)
  (OpenMP support for the [clang](https://clang.llvm.org/) C compiler)
- [mkl-devel](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html)
  (fast numerical computation)

## Usage

Consider `n` independent and non-identically distributed random variables
and be `p` a list/numpy array of the corresponding Bernoulli success
probabilities. In order to create the Poisson binomial distributions, use
```python
from cpoibin import PoiBin
pb = PoiBin(p)
```
Let `x` be a list/numpy array containing different numbers of successes.
Use the following methods to obtain the corresponding quantities:

- Probability mass function
```python
pb.pmf(x)
```
- Cumulative distribution function
```python
pb.cdf(x)
```
- p-values for right tailed tests
```python
pb.pval(x)
```

All three methods accept single integers as well as lists/numpy arrays of
integers. Note that each element of `x`, `x[i]`, must be between 0 and
`len(p)`, inclusive on both ends (there are between 0 and `n` successes).

## Testing

Testing has been implemented using the
`pytest` module. To run the tests, execute
```
$ pytest test_poibin.py
```
in the command line. For verbose mode, use

```
$ pytest -v test_poibin.py
```

## References

[[1] Yili Hong, On computing the distribution function for the Poisson binomial
distribution, Computational Statistics & Data Analysis, Volume 59, March 2013,
pages 41-51, ISSN 0167-9473](https://doi.org/10.1016/j.csda.2012.10.006)

[[2] L. A. Belfore, "An O(n/spl middot/(log/sub 2/(n))/sup 2/) algorithm for
computing the reliability of k-out-of-n:G and k-to-l-out-of-n:G systems," in
IEEE Transactions on Reliability, vol. 44, no. 1, pp. 132-136, March 1995,
doi: 10.1109/24.376535.](https://doi.org/10.1109/24.376535)

