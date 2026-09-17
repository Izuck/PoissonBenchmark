"""Adapters to existing numerical packages; no custom PDE discretization."""
import os
import sys
from pathlib import Path
from time import perf_counter

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(Path(__file__).parent / '.mplconfig')
binpath = Path(sys.prefix) / 'Library' / 'bin'
if binpath.exists():
    os.environ['PATH'] = str(binpath) + os.pathsep + os.environ['PATH']
    _dll_handle = os.add_dll_directory(str(binpath))

import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator

MODES = ((1, 1), (1, 2), (2, 1), (2, 3))


def field(x, y, coefficients, source=False):
    return sum(a * (np.pi**2 * (m*m+n*n) if source else 1.) *
               np.sin(m*np.pi*x) * np.sin(n*np.pi*y)
               for a, (m, n) in zip(coefficients, MODES))


def grid(size):
    axis = (np.arange(size, dtype=np.float64) + .5) / size
    return np.meshgrid(axis, axis, indexing='xy')


class Base:
    def sample(self, size=512):
        # Use actual element connectivity, not a new Delaunay triangulation.
        if not hasattr(self, 'tri'):
            self.tri = Triangulation(self.xy[:, 0], self.xy[:, 1], self.cells)
        x, y = grid(size)
        values = LinearTriInterpolator(self.tri, self.u)(x, y)
        if np.ma.getmaskarray(values).any():
            raise ValueError('Sampling produced points outside the mesh')
        return np.asarray(values, dtype=np.float64)

    def stats(self):
        return dict(nodes=len(self.xy), cells=len(self.cells),
                    free_dofs=int((~self.boundary).sum()),
                    boundary_max_abs_error=float(np.max(np.abs(self.u[self.boundary]))),
                    center_value=float(self.u[np.argmin(np.sum((self.xy-.5)**2, axis=1))]),
                    nodal_min=float(self.u.min()), nodal_max=float(self.u.max()))

    def close(self):
        pass


class ScikitFEM(Base):
    backend = 'SciPy SuperLU (splu, COLAMD)'
    reuse = 'mesh, P1 basis, condensed stiffness, SuperLU factorization, sampling triangulation; RHS reassembled'

    def __init__(self, n):
        import skfem as fem
        from skfem.models.poisson import laplace
        from scipy.sparse.linalg import splu
        self.fem = fem
        t = perf_counter()
        self.mesh = fem.MeshTri.init_tensor(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
        self.xy, self.cells = self.mesh.p.T, self.mesh.t.T
        self.times = {'mesh_s': perf_counter()-t}
        t = perf_counter()
        self.basis = fem.Basis(self.mesh, fem.ElementTriP1(), intorder=6)
        self.boundary = np.zeros(self.basis.N, dtype=bool)
        self.boundary[self.basis.get_dofs().all()] = True
        self.interior = np.flatnonzero(~self.boundary)
        self.A = fem.asm(laplace, self.basis)[self.interior][:, self.interior].tocsc()
        self.times['assembly_s'] = perf_counter()-t
        t = perf_counter()
        self.factor = splu(self.A, permc_spec='COLAMD')
        self.times['solver_setup_s'] = perf_counter()-t

    def solve(self, coefficients, tolerance=1e-10):
        t = perf_counter()
        @self.fem.LinearForm
        def rhs(v, w):
            return field(w.x[0], w.x[1], coefficients, source=True)*v
        b = self.fem.asm(rhs, self.basis)[self.interior]
        assembly = perf_counter()-t
        t = perf_counter()
        self.u = np.zeros(len(self.xy), dtype=np.float64)
        self.u[self.interior] = self.factor.solve(b)
        elapsed = perf_counter()-t
        self.b = b
        return dict(assembly_s=assembly, solve_s=elapsed,
                    iterations=1,
                    solver_info={'method': 'direct LU', 'tolerance': 'not applicable'})

    def residual(self):
        return float(np.linalg.norm(self.A @ self.u[self.interior]-self.b) / np.linalg.norm(self.b))


class Devsim(Base):
    backend = 'DEVSIM MKL PARDISO direct, DC Newton'
    reuse = 'mesh, equations, source model, sampling triangulation; solve reassembles/refactorizes; no claimed numeric LU reuse'

    def __init__(self, n):
        import devsim as d
        self.d = d
        self.kw = dict(device='bench', region='bulk')
        t = perf_counter()
        from skfem import MeshTri
        mesh = MeshTri.init_tensor(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
        coords = np.column_stack((mesh.p.T, np.zeros(mesh.p.shape[1])))
        triangles = np.column_stack((np.full(mesh.t.shape[1], 2), np.zeros(mesh.t.shape[1], dtype=int), mesh.t.T))
        edges = mesh.facets[:, mesh.boundary_facets()].T
        segments = np.column_stack((np.ones(len(edges), dtype=int), np.ones(len(edges), dtype=int), edges))
        d.create_gmsh_mesh(mesh='mesh', coordinates=coords.ravel().tolist(),
                           elements=np.concatenate((triangles.ravel(), segments.ravel())).tolist(),
                           physical_names=['bulk', 'boundary'])
        d.add_gmsh_region(mesh='mesh', gmsh_name='bulk', region='bulk', material='dimensionless')
        d.add_gmsh_contact(mesh='mesh', gmsh_name='boundary', name='boundary', region='bulk', material='metal')
        d.finalize_mesh(mesh='mesh')
        d.create_device(mesh='mesh', device='bench')
        self.xy = np.column_stack([d.get_node_model_values(**self.kw, name=v) for v in ('x','y')])
        self.cells = np.array(d.get_element_node_list(**self.kw), dtype=np.int32)
        self.boundary = np.any(np.isclose(self.xy, 0) | np.isclose(self.xy, 1), axis=1)
        self.times = {'mesh_s':perf_counter()-t}
        t = perf_counter()
        d.node_solution(**self.kw, name='u')
        d.edge_from_node_model(**self.kw, node_model='u')
        d.edge_model(**self.kw, name='flux', equation='(u@n0-u@n1)*EdgeInverseLength')
        d.edge_model(**self.kw, name='flux:u@n0', equation='EdgeInverseLength')
        d.edge_model(**self.kw, name='flux:u@n1', equation='-EdgeInverseLength')
        # Outward negative-gradient flux gives -laplacian(u); node term is -f.
        d.node_solution(**self.kw, name='source')
        d.node_model(**self.kw, name='source:u', equation='0')
        d.equation(**self.kw, name='Poisson', variable_name='u', edge_model='flux', node_model='source')
        for contact in d.get_contact_list(device='bench'):
            d.contact_node_model(device='bench', contact=contact, name='bc', equation='u')
            d.contact_node_model(device='bench', contact=contact, name='bc:u', equation='1')
            d.contact_equation(device='bench', contact=contact, name='Poisson', node_model='bc')
        self.times.update(assembly_s=perf_counter()-t, solver_setup_s=None)

    def solve(self, coefficients, tolerance=1e-10):
        t = perf_counter()
        self.d.set_node_values(**self.kw, name='source',
                               values=-field(self.xy[:,0], self.xy[:,1], coefficients, source=True))
        self.d.set_node_values(**self.kw, name='u', values=np.zeros(len(self.xy)))
        assembly = perf_counter()-t
        t = perf_counter()
        info = self.d.solve(type='dc', solver_type='direct', absolute_error=tolerance,
                            relative_error=tolerance, maximum_iterations=20, info=True)
        elapsed = perf_counter()-t
        self.u = np.array(self.d.get_node_model_values(**self.kw, name='u'))
        if not info.get('converged', False):
            raise RuntimeError(f'DEVSIM failed: {info}')
        return dict(assembly_s=assembly, solve_s=elapsed, solver_info=info,
                    iterations=len(info.get('iterations', [])))

    def close(self):
        self.d.delete_device(device='bench')
        self.d.delete_mesh(mesh='mesh')


ADAPTERS = {'devsim': Devsim, 'scikit-fem': ScikitFEM}
