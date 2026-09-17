"""Simple charge-free MOS-capacitor cross-section, dimensionless coordinates.

Width and height: 100 nm. Silicon: y=0..0.9; oxide: y=0.9..1.
Relative permittivities: 11.7 and 3.9. Bottom/gate voltages are Dirichlet;
sidewalls are natural zero flux. Metal is represented by a boundary only.
"""
from time import perf_counter
from adapters import Base, ScikitFEM, grid, np
from problems import MOSCAP,exact_moscap

INTERFACE=MOSCAP['interface_y_normalized']
EPS_SI=MOSCAP['materials']['silicon_relative_permittivity']
EPS_OX=MOSCAP['materials']['oxide_relative_permittivity']


def field(x,y,coefficients,source=False):
    return exact_moscap(x,y,coefficients)


def make_mesh(n):
    from skfem import MeshTri
    y=np.concatenate((np.linspace(0,INTERFACE,3*n//4+1),
                      np.linspace(INTERFACE,1,n//4+1)[1:]))
    return MeshTri.init_tensor(np.linspace(0,1,n+1),y)


class CapBase(Base):
    def stats(self):
        stats=super().stats()
        exact=field(self.xy[:,0],self.xy[:,1],self.coefficients)
        stats['boundary_max_abs_error']=float(np.max(np.abs(self.u[self.boundary]-exact[self.boundary])))
        stats['nodal_max_abs_error']=float(np.max(np.abs(self.u-exact)))
        return stats


class CapFEM(CapBase,ScikitFEM):
    backend=ScikitFEM.backend
    reuse='mesh, P1 basis, reduced dielectric stiffness, SuperLU factorization, sampling triangulation; Dirichlet RHS updated'

    def __init__(self,n):
        import skfem as fem
        from skfem.helpers import dot,grad
        from scipy.sparse.linalg import splu
        t=perf_counter()
        self.mesh=make_mesh(n)
        self.xy,self.cells=self.mesh.p.T,self.mesh.t.T
        self.times=dict(mesh_s=perf_counter()-t)
        t=perf_counter()
        self.basis=fem.Basis(self.mesh,fem.ElementTriP1(),intorder=2)
        self.boundary=np.isclose(self.xy[:,1],0)|np.isclose(self.xy[:,1],1)
        self.interior=np.flatnonzero(~self.boundary)
        self.fixed=np.flatnonzero(self.boundary)
        @fem.BilinearForm
        def dielectric(u,v,w):
            eps=np.where(w.x[1]<INTERFACE,EPS_SI,EPS_OX)
            return eps*dot(grad(u),grad(v))
        full=fem.asm(dielectric,self.basis)
        self.A=full[self.interior][:,self.interior].tocsc()
        self.Aib=full[self.interior][:,self.fixed].tocsr()
        self.times['assembly_s']=perf_counter()-t
        t=perf_counter()
        self.factor=splu(self.A,permc_spec='COLAMD')
        self.times['solver_setup_s']=perf_counter()-t

    def solve(self,coefficients,tolerance=1e-10):
        self.coefficients=coefficients
        t=perf_counter()
        gate,bottom=coefficients
        self.u=np.where(np.isclose(self.xy[:,1],1),gate,bottom).astype(float)
        self.u[self.interior]=0.
        self.b= -self.Aib@self.u[self.fixed]
        assembly=perf_counter()-t
        t=perf_counter()
        self.u[self.interior]=self.factor.solve(self.b)
        return dict(assembly_s=assembly,solve_s=perf_counter()-t,iterations=1,
                    solver_info={'method':'direct LU','tolerance':'not applicable'})


class CapDevsim(CapBase):
    backend='DEVSIM MKL PARDISO direct, DC Newton, two dielectric regions'
    reuse='two-region mesh, equations, interface, sampling triangulation; PARDISO numerical factorization not explicitly reused'

    def __init__(self,n):
        import devsim as d
        self.d=d
        t=perf_counter()
        mesh=make_mesh(n)
        coords=np.column_stack((mesh.p.T,np.zeros(mesh.p.shape[1])))
        tri=mesh.t.T
        groups=(mesh.p[1,tri].mean(axis=1)>INTERFACE).astype(int)
        elements=np.column_stack((np.full(len(tri),2),groups,tri)).ravel().tolist()
        facets=mesh.facets.T
        for group,yval in [(2,0.),(3,1.),(4,INTERFACE)]:
            edges=facets[np.all(np.isclose(mesh.p[1,facets],yval),axis=1)]
            elements.extend(np.column_stack((np.ones(len(edges),dtype=int),np.full(len(edges),group),edges)).ravel().tolist())
        d.create_gmsh_mesh(mesh='capmesh',coordinates=coords.ravel().tolist(),elements=elements,
                           physical_names=['silicon','oxide','bottom','gate','interface'])
        for region in ('silicon','oxide'):
            d.add_gmsh_region(mesh='capmesh',gmsh_name=region,region=region,material=region)
        for contact,region in [('bottom','silicon'),('gate','oxide')]:
            d.add_gmsh_contact(mesh='capmesh',gmsh_name=contact,name=contact,region=region,material='metal')
        d.add_gmsh_interface(mesh='capmesh',gmsh_name='interface',name='interface',region0='silicon',region1='oxide')
        d.finalize_mesh(mesh='capmesh')
        d.create_device(mesh='capmesh',device='cap')
        # Map regional nodes (duplicated at the material interface) back to a common mesh.
        self.xy,self.cells=mesh.p.T,mesh.t.T
        lookup={tuple(x):i for i,x in enumerate(self.xy)}
        self.region_indices={}
        for region in ('silicon','oxide'):
            rxy=np.column_stack([d.get_node_model_values(device='cap',region=region,name=v) for v in ('x','y')])
            self.region_indices[region]=np.array([lookup[tuple(x)] for x in rxy])
        self.boundary=np.isclose(self.xy[:,1],0)|np.isclose(self.xy[:,1],1)
        self.times=dict(mesh_s=perf_counter()-t)
        t=perf_counter()
        for region,eps in [('silicon',EPS_SI),('oxide',EPS_OX)]:
            kw=dict(device='cap',region=region)
            d.set_parameter(**kw,name='eps',value=eps)
            d.node_solution(**kw,name='u')
            d.edge_from_node_model(**kw,node_model='u')
            d.edge_model(**kw,name='flux',equation='eps*(u@n0-u@n1)*EdgeInverseLength')
            d.edge_model(**kw,name='flux:u@n0',equation='eps*EdgeInverseLength')
            d.edge_model(**kw,name='flux:u@n1',equation='-eps*EdgeInverseLength')
            d.equation(**kw,name='Poisson',variable_name='u',edge_model='flux')
        for contact,region in [('bottom','silicon'),('gate','oxide')]:
            d.set_parameter(device='cap',region=region,name='bias',value=0.)
            d.contact_node_model(device='cap',contact=contact,name='bc',equation='u-bias')
            d.contact_node_model(device='cap',contact=contact,name='bc:u',equation='1')
            d.contact_equation(device='cap',contact=contact,name='Poisson',node_model='bc')
        for name,expr in [('continuity','u@r0-u@r1'),('continuity:u@r0','1'),('continuity:u@r1','-1')]:
            d.interface_model(device='cap',interface='interface',name=name,equation=expr)
        d.interface_equation(device='cap',interface='interface',name='Poisson',interface_model='continuity',type='continuous')
        self.times.update(assembly_s=perf_counter()-t,solver_setup_s=None)

    def solve(self,coefficients,tolerance=1e-10):
        self.coefficients=coefficients
        gate,bottom=coefficients
        t=perf_counter()
        for region,bias in [('silicon',bottom),('oxide',gate)]:
            self.d.set_parameter(device='cap',region=region,name='bias',value=float(bias))
            self.d.set_node_values(device='cap',region=region,name='u',values=np.zeros(len(self.region_indices[region])))
        assembly=perf_counter()-t
        t=perf_counter()
        info=self.d.solve(type='dc',solver_type='direct',absolute_error=tolerance,relative_error=tolerance,
                          maximum_iterations=20,info=True)
        elapsed=perf_counter()-t
        if not info.get('converged',False):
            raise RuntimeError(str(info))
        self.u=np.zeros(len(self.xy))
        counts=np.zeros(len(self.xy))
        region_values={}
        for region,indices in self.region_indices.items():
            values=np.array(self.d.get_node_model_values(device='cap',region=region,name='u'))
            region_values[region]=dict(zip(indices,values))
            self.u[indices]+=values
            counts[indices]+=1
        interface=np.flatnonzero(counts==2)
        self.interface_jump=max(abs(region_values['silicon'][i]-region_values['oxide'][i]) for i in interface)
        if self.interface_jump > 1e-10:
            raise ValueError('Discontinuous potential at the material interface')
        self.u/=counts
        return dict(assembly_s=assembly,solve_s=elapsed,iterations=len(info['iterations']),solver_info=info,
                    interface_max_jump=self.interface_jump)

    def close(self):
        self.d.delete_device(device='cap')
        self.d.delete_mesh(mesh='capmesh')


ADAPTERS={'devsim':CapDevsim,'scikit-fem':CapFEM}
