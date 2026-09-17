"""Run both analytical cases at N=256: python devsim_benchmark.py."""
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
# Make a project-local MKL installation visible on Windows.
binpath = Path(sys.prefix) / 'Library' / 'bin'
if sys.platform == 'win32' and binpath.exists():
    os.environ['PATH'] = str(binpath) + os.pathsep + os.environ['PATH']
    dll_handle = os.add_dll_directory(str(binpath))

import devsim as d
import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshTri, Basis, ElementTriP1  # Mesh and error integration only.

N = 256
REPEATS = 5


def exact(x, y, regular):
    if regular:
        return (np.minimum(y, .9) / 11.7 + np.maximum(y - .9, 0) / 3.9) / (.9 / 11.7 + .1 / 3.9)
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def solve(regular):
    start = perf_counter()
    axis = np.linspace(0, 1, N + 1)
    y = np.r_[np.linspace(0, .9, 3 * N // 4 + 1),
              np.linspace(.9, 1, N // 4 + 1)[1:]] if regular else axis
    mesh = MeshTri.init_tensor(axis, y)
    triangles, edges = mesh.t.T, mesh.facets.T
    regions = [('silicon', 11.7), ('oxide', 3.9)] if regular else [('bulk', 1.)]
    contacts = [('bottom', 'silicon', 0.), ('gate', 'oxide', 1.)] if regular else [('boundary', 'bulk', 0.)]

    # Import the same triangular geometry as the scikit-fem test.
    groups = (mesh.p[1, triangles].mean(axis=1) > .9).astype(int) if regular else np.zeros(len(triangles), int)
    elements = np.column_stack((np.full(len(triangles), 2), groups, triangles)).ravel().tolist()
    names = [name for name, _ in regions] + [name for name, _, _ in contacts]
    if regular:
        names.append('interface')
        edge_groups = [(2, edges[np.all(np.isclose(mesh.p[1, edges], 0), axis=1)]),
                       (3, edges[np.all(np.isclose(mesh.p[1, edges], 1), axis=1)]),
                       (4, edges[np.all(np.isclose(mesh.p[1, edges], .9), axis=1)])]
    else:
        edge_groups = [(1, mesh.facets[:, mesh.boundary_facets()].T)]
    for group, selected in edge_groups:
        elements.extend(np.column_stack((np.ones(len(selected), int),
                                         np.full(len(selected), group), selected)).ravel().tolist())
    coords = np.column_stack((mesh.p.T, np.zeros(mesh.p.shape[1])))
    d.create_gmsh_mesh(mesh='mesh', coordinates=coords.ravel().tolist(), elements=elements, physical_names=names)
    for region, _ in regions:
        d.add_gmsh_region(mesh='mesh', gmsh_name=region, region=region, material=region)
    for contact, region, _ in contacts:
        d.add_gmsh_contact(mesh='mesh', gmsh_name=contact, name=contact, region=region, material='metal')
    if regular:
        d.add_gmsh_interface(mesh='mesh', gmsh_name='interface', name='interface', region0='silicon', region1='oxide')
    d.finalize_mesh(mesh='mesh')
    d.create_device(mesh='mesh', device='device')

    # -div(epsilon grad(u)) = source. Zero charge for regular, sine source otherwise.
    for region, epsilon in regions:
        kw = dict(device='device', region=region)
        d.set_parameter(**kw, name='eps', value=epsilon)
        d.node_solution(**kw, name='u')
        d.edge_from_node_model(**kw, node_model='u')
        d.edge_model(**kw, name='flux', equation='eps*(u@n0-u@n1)*EdgeInverseLength')
        d.edge_model(**kw, name='flux:u@n0', equation='eps*EdgeInverseLength')
        d.edge_model(**kw, name='flux:u@n1', equation='-eps*EdgeInverseLength')
        d.node_solution(**kw, name='source')
        x, y = [np.array(d.get_node_model_values(**kw, name=v)) for v in ('x', 'y')]
        charge = np.zeros(len(x)) if regular else -2 * np.pi**2 * exact(x, y, False)
        d.set_node_values(**kw, name='source', values=charge)
        d.node_model(**kw, name='source:u', equation='0')
        d.equation(**kw, name='Poisson', variable_name='u', edge_model='flux', node_model='source')
    for contact, region, bias in contacts:
        d.set_parameter(device='device', region=region, name='bias', value=bias)
        kw = dict(device='device', contact=contact)
        d.contact_node_model(**kw, name='bc', equation='u-bias')
        d.contact_node_model(**kw, name='bc:u', equation='1')
        d.contact_equation(**kw, name='Poisson', node_model='bc')
    if regular:
        kw = dict(device='device', interface='interface')
        for name, expr in [('continuity', 'u@r0-u@r1'), ('continuity:u@r0', '1'), ('continuity:u@r1', '-1')]:
            d.interface_model(**kw, name=name, equation=expr)
        d.interface_equation(**kw, name='Poisson', interface_model='continuity', type='continuous')
    info = d.solve(type='dc', solver_type='direct', absolute_error=1e-10,
                   relative_error=1e-10, maximum_iterations=20, info=True)
    fields = []
    for region, _ in regions:
        kw = dict(device='device', region=region)
        xy = np.array([d.get_node_model_values(**kw, name=v) for v in ('x', 'y')])
        cells = np.array(d.get_element_node_list(**kw), dtype=int).T
        voltage = np.array(d.get_node_model_values(**kw, name='u'))
        fields.append((xy, cells, voltage))
    seconds = perf_counter() - start

    # Integrate each region separately; the shared interface has zero area.
    numerator = denominator = boundary_error = 0.
    for xy, cells, voltage in fields:
        check = Basis(MeshTri(xy, cells), ElementTriP1(), intorder=6)
        truth = exact(*check.global_coordinates(), regular)
        numerator += np.sum((check.interpolate(voltage) - truth)**2 * check.dx)
        denominator += np.sum(truth**2 * check.dx)
        x, y = xy
        boundary = np.isclose(y, 0) | np.isclose(y, 1)
        if not regular:
            boundary |= np.isclose(x, 0) | np.isclose(x, 1)
        boundary_error = max(boundary_error, np.max(np.abs(voltage[boundary] - exact(x[boundary], y[boundary], regular))))
    error = np.sqrt(numerator / denominator)
    d.delete_device(device='device')
    d.delete_mesh(mesh='mesh')
    if not info['converged'] or not np.isfinite(error) or error >= (1e-10 if regular else 1e-3) or boundary_error >= 1e-10:
        raise ValueError('Solver convergence, analytical, or boundary check failed')
    return seconds, error, fields


if __name__ == '__main__':
    rows = []
    # Keep verbose solver diagnostics in one local log instead of the results table.
    with Path(__file__).with_suffix('.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        for name, regular in [('Regular', True), ('Sine', False)]:
            solve(regular)  # Excluded warm-up.
            results = [solve(regular) for _ in range(REPEATS)]
            times, errors, fields = zip(*results)
            rows.append((name, regular, times, max(errors), fields[-1]))
    print(f'DEVSIM {version("devsim")} | MKL {version("mkl")} | N={N} | one CPU thread')
    print('Time = mesh + assembly + factorization + solve + field extraction; excludes imports and checking.')
    print(f'{"Case":<12} {"Median (s)":>12} {"Relative L2":>14}  Measured times (s)')
    for name, regular, times, error, fields in rows:
        print(f'{name:<12} {median(times):>12.3f} {error:>14.3e}  '
              + ', '.join(f'{t:.3f}' for t in times))
    print('Both cases passed. Solver diagnostics: devsim_benchmark.log')

    # Plot only after all timings finish: a vertical slice through x=0.5.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    y = np.linspace(0, 1, 501)
    for ax, (name, regular, times, error, fields) in zip(axes, rows):
        ax.plot(y, exact(.5, y, regular), color='black', label='Exact')
        for i, (xy, cells, voltage) in enumerate(fields):
            center = np.flatnonzero(np.isclose(xy[0], .5))
            center = center[np.argsort(xy[1, center])]
            ax.plot(xy[1, center], voltage[center], '.', color='tab:orange',
                    markevery=8, label='DEVSIM' if i == 0 else None)
        ax.set(title=f'{name}: {median(times):.3f} s; relative L2 = {error:.2e}',
               xlabel='Height / total height', ylabel='Potential (V for regular)')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle(f'DEVSIM: N={N}, centerline x=0.5')
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=150)
    plt.show()
