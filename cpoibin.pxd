# expose certain low-level functions for testing and cython use

from fftw cimport fftw_plan

cdef double direct_dot(int n, double *x, double *y)

cdef double blas_dot(int n, double *x, double *y)

cdef (double (*)(int n, double *x, double *y)) get_dot(int n)

cdef void direct_convolve(int n, double *x, int m, double *y)

cdef void elementwise_product(int n, double *x, const double *y)

cdef void fft_convolve(int n, double *x, int m, double *y,
                       fftw_plan plan_forward, fftw_plan plan_backward,
                       double *temp)

cdef void convolve(int n, double *x, int m, double *y,
                   fftw_plan plan_forward, fftw_plan plan_backward,
                   double *temp)

