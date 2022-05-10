import time
import numpy as np
from poibin import PoiBin
from cpoibin import PoiBin as cPoiBin

TRIALS = 10

# set seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000]
    for size in sizes:
        poibin_time, cpoibin_time = 0, 0
        for trial in range(TRIALS):
            # uniform in [0, 1]
            probs = rng.random(size)

            start = time.time()
            pb = PoiBin(probs)
            poibin_pdf = pb.pmf(np.arange(size + 1))
            poibin_time += time.time() - start

            start = time.time()
            cpb = cPoiBin(probs)
            cpoibin_pdf = cpb.pmf(np.arange(size + 1))
            cpoibin_time += time.time() - start

            assert np.allclose(poibin_pdf, cpoibin_pdf), "methods don't agree"

        poibin_time, cpoibin_time = poibin_time/TRIALS, cpoibin_time/TRIALS
        print(f"size {size:5}: poibin {poibin_time:.3e} c {cpoibin_time:.3e} "
              f"ratio {poibin_time/cpoibin_time:.3f}")

