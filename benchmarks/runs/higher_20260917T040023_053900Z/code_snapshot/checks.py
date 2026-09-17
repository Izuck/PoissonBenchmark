"""Compare any solver's voltage with the known formula; never timed as solving."""
import numpy as np


def relative_error(values,exact):
    return float(np.linalg.norm(values-exact)/np.linalg.norm(exact))


def integrated_error(adapter,reference,coefficients,order=6):
    """Integrate squared error over actual triangles, not the coarse export grid."""
    from skfem import MeshTri,Basis,ElementTriP1
    basis=Basis(MeshTri(adapter.xy.T,adapter.cells.T),ElementTriP1(),intorder=order)
    x,y=basis.global_coordinates()
    exact=reference(x,y,coefficients)
    numerical=basis.interpolate(adapter.u)
    return float(np.sqrt(np.sum((numerical-exact)**2*basis.dx)/np.sum(exact**2*basis.dx)))
