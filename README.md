# Two simple Poisson benchmarks

**Two scripts. Two cases. N = 256.** Each script prints a results table and uses Matplotlib to plot the calculated voltage against the exact answer.

- [scikit_benchmark.py](scikit_benchmark.py): scikit-fem linear triangular elements, solved with SciPy SuperLU.
- [devsim_benchmark.py](devsim_benchmark.py): DEVSIM, solved with MKL PARDISO. It uses scikit-fem only for mesh generation and error integration.

## Cases

**Regular:** a 100 nm square capacitor, with 90 nm silicon below 10 nm oxide. Relative permittivities are 11.7 and 3.9, top voltage is 1 V, bottom is 0 V, and sides are insulated. There is no charge. The exact voltage is linear within each material, reaching 0.75 V at the interface.

**Sine:** a unit square with zero boundary voltage and source `2*pi^2*sin(pi*x)*sin(pi*y)`. The exact answer is `sin(pi*x)*sin(pi*y)`.

## Run

Tested with Python 3.12.14 on 64-bit Windows. Create a local environment once:

```powershell
py -3.12 -m venv .venv
& .venv/Scripts/python.exe -m pip install numpy==2.5.3 scipy==1.18.1 scikit-fem==12.0.2 devsim==2.11.0 mkl==2025.3.0 matplotlib==3.11.2
```

Run either script:

```powershell
& .venv/Scripts/python.exe scikit_benchmark.py
& .venv/Scripts/python.exe devsim_benchmark.py
```

Each script runs both cases, shows one figure with two panels, and saves that figure next to the script. The panels show a vertical slice through the middle of the domain. Runtime and full-domain error appear in the titles. DEVSIM saves its verbose diagnostics to `devsim_benchmark.log`.

## Reading the code and results

Both files follow the same sequence: `exact()` gives the known answer; `solve()` builds and solves one case, then checks it; the bottom loop repeats each case and produces the table and plot. No shared project modules or separate plotting script are needed.

Each case gets one excluded warm-up and five independent solves. The table prints all five times, their median, and the worst relative L2 error. Each solve rebuilds the mesh and factorization. One CPU computation thread is requested through process-local environment variables.

**Timed:** mesh creation, equation assembly, factorization, solving, and obtaining nodal voltage. **Excluded:** imports, analytical checks, cleanup, and plotting. There are no voltage-array exports, so these timings differ in scope from the earlier benchmark.

**Checked:** full-domain relative error, integrated over triangles with order-6 quadrature, must be below `1e-10` for regular and `1e-3` for sine. Boundary error must be below `1e-10`. The plot is just a centerline view; the accuracy check covers the whole domain. These are baseline thresholds, not final solver requirements. A single mesh size cannot demonstrate convergence or general surrogate accuracy.

## Example results

Measured on an Intel Core Ultra 9 285H, 31.6 GiB usable RAM, Windows 11; CPU only, no GPU. Median seconds from five repetitions at N=256:

| Tool | Case | Seconds | Relative L2 error |
|---|---|---:|---:|
| scikit-fem | Regular | 0.388 | 4.709e-13 |
| scikit-fem | Sine | 0.700 | 4.226e-5 |
| DEVSIM | Regular | 2.244 | 1.742e-16 |
| DEVSIM | Sine | 2.299 | 2.174e-5 |

Both tools passed both cases. Background activity and filesystem caches were not controlled; timings vary between runs. The previous framework and measurements remain in Git history.
