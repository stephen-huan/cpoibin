# https://github.com/pyFFTW/pyFFTW/blob/master/pyfftw/pyfftw.pxd

cdef extern from "fftw3.h":

    ctypedef struct fftw_plan_struct:
        pass

    ctypedef fftw_plan_struct *fftw_plan

    fftw_plan fftw_plan_r2r_1d(int n, double *in_, double *out,
                               int kind, unsigned flags)

    void fftw_execute_r2r(const fftw_plan p, double *in_, double *out)

    void fftw_destroy_plan(fftw_plan p)
    void fftw_forget_wisdom()
    void fftw_cleanup()

    void fftw_set_timelimit(double t)

    # threading

    void fftw_plan_with_nthreads(int nthreads)
    int fftw_init_threads()
    void fftw_cleanup_threads()

    # wisdom

    int fftw_export_wisdom_to_filename(const char *filename)
    char *fftw_export_wisdom_to_string();
    int fftw_import_system_wisdom()
    int fftw_import_wisdom_from_filename(const char *filename)

    char *fftw_sprint_plan(const fftw_plan p)

    # memory

    void *fftw_malloc(size_t n)
    double *fftw_alloc_real(size_t n)
    void fftw_free(void *p)

# directions
cdef enum:
    FFTW_FORWARD = -1
    FFTW_BACKWARD = 1

# r2r kinds
cdef enum:
    FFTW_R2HC = 0
    FFTW_HC2R = 1
    FFTW_DHT = 2
    FFTW_REDFT00 = 3
    FFTW_REDFT01 = 4
    FFTW_REDFT10 = 5
    FFTW_REDFT11 = 6
    FFTW_RODFT00 = 7
    FFTW_RODFT01 = 8
    FFTW_RODFT10 = 9
    FFTW_RODFT11 = 10

# documented flags
cdef enum:
    FFTW_MEASURE = 0
    FFTW_DESTROY_INPUT = 1
    FFTW_UNALIGNED = 2
    FFTW_CONSERVE_MEMORY = 4
    FFTW_EXHAUSTIVE = 8
    FFTW_PRESERVE_INPUT = 16
    FFTW_PATIENT = 32
    FFTW_ESTIMATE = 64
    FFTW_WISDOM_ONLY = 2097152

