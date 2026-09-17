"""Run both analytical cases at N=256: python scikit_benchmark.py."""
import os
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

# Set these before importing numerical libraries; only this process is affected.
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm, asm
from skfem.helpers import dot, grad

N = 256
REPEATS = 5


def exact(x, y, regular):
    if regular:
        return (np.minimum(y, .9) / 11.7 + np.maximum(y - .9, 0) / 3.9) / (.9 / 11.7 + .1 / 3.9)
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def solve(regular):
    start = perf_counter()
    # Regular: 90 nm silicon + 10 nm oxide. Sine: uniform unit square.
    axis = np.linspace(0, 1, N + 1)
    y = np.r_[np.linspace(0, .9, 3 * N // 4 + 1),
              np.linspace(.9, 1, N // 4 + 1)[1:]] if regular else axis
    mesh = MeshTri.init_tensor(axis, y)
    basis = Basis(mesh, ElementTriP1(), intorder=2 if regular else 6)

    @BilinearForm
    def stiffness(u, v, w):
        epsilon = np.where(w.x[1] < .9, 11.7, 3.9) if regular else 1.
        return epsilon * dot(grad(u), grad(v))

    @LinearForm
    def source(v, w):
        return 2 * np.pi**2 * exact(*w.x, False) * v

    matrix = asm(stiffness, basis)
    rhs = np.zeros(basis.N) if regular else asm(source, basis)
    x, y = mesh.p
    boundary = np.isclose(y, 0) | np.isclose(y, 1)
    if not regular:
        boundary |= np.isclose(x, 0) | np.isclose(x, 1)
    fixed, free = np.flatnonzero(boundary), np.flatnonzero(~boundary)
    voltage = np.zeros(basis.N)
    if regular:
        voltage[np.isclose(y, 1)] = 1.
    # Eliminate fixed boundary values, then factor and solve from scratch.
    reduced = matrix[free][:, free].tocsc()
    load = rhs[free] - matrix[free][:, fixed] @ voltage[fixed]
    voltage[free] = splu(reduced, permc_spec='COLAMD').solve(load)
    seconds = perf_counter() - start

    # Integrate error over triangles, outside the timed calculation.
    check = Basis(mesh, ElementTriP1(), intorder=6)
    truth = exact(*check.global_coordinates(), regular)
    error = np.sqrt(np.sum((check.interpolate(voltage) - truth)**2 * check.dx)
                    / np.sum(truth**2 * check.dx))
    boundary_error = np.max(np.abs(voltage[fixed] - exact(x[fixed], y[fixed], regular)))
    if not np.isfinite(error) or error >= (1e-10 if regular else 1e-3) or boundary_error >= 1e-10:
        raise ValueError('Analytical or boundary check failed')
    return seconds, error, (mesh.p, voltage)


if __name__ == '__main__':
    print(f'scikit-fem {version("scikit-fem")} | SciPy {version("scipy")} | N={N} | one CPU thread')
    print('Time = mesh + assembly + factorization + solve; excludes imports and checking.')
    print(f'{"Case":<12} {"Median (s)":>12} {"Relative L2":>14}  Measured times (s)')
    rows = []
    for name, regular in [('Regular', True), ('Sine', False)]:
        solve(regular)  # Excluded warm-up.
        results = [solve(regular) for _ in range(REPEATS)]
        times, errors, fields = zip(*results)
        rows.append((name, regular, times, max(errors), fields[-1]))
        print(f'{name:<12} {median(times):>12.3f} {max(errors):>14.3e}  '
              + ', '.join(f'{t:.3f}' for t in times))
    print('Both cases passed.')

    # Plot only after all timings finish: a vertical slice through x=0.5.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    y = np.linspace(0, 1, 501)
    for ax, (name, regular, times, error, (xy, voltage)) in zip(axes, rows):
        ax.plot(y, exact(.5, y, regular), color='black', label='Exact')
        center = np.flatnonzero(np.isclose(xy[0], .5))
        center = center[np.argsort(xy[1, center])]
        ax.plot(xy[1, center], voltage[center], '.', color='tab:blue',
                markevery=8, label='scikit-fem')
        ax.set(title=f'{name}: {median(times):.3f} s; relative L2 = {error:.2e}',
               xlabel='Height / total height', ylabel='Potential (V for regular)')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle(f'scikit-fem: N={N}, centerline x=0.5')
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=150)
    plt.show()
