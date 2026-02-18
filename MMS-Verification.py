# Verification using Method of Manufactured Solutions (MMS)
# Ensures correctness of finite element implementation.



from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl
from basix.ufl import element, mixed_element



# Convergence study

def run_convergence_study(N_values, nu_val=1e-3, D_val=5e-4):

    errors_u = []
    errors_p = []
    errors_c = []
    h_values = []

    for N in N_values:

        # Mesh
        domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)

        # Function spaces (Taylor–Hood)
        degree = 1
        P2 = element("Lagrange", domain.basix_cell(), degree + 1, shape=(2,))
        P1 = element("Lagrange", domain.basix_cell(), degree)

        TH = mixed_element([P2, P1])
        W = fem.functionspace(domain, TH)
        C = fem.functionspace(domain, ("Lagrange", degree))

        # Coordinates + parameters
        x = ufl.SpatialCoordinate(domain)
        nu = fem.Constant(domain, default_scalar_type(nu_val))
        D = fem.Constant(domain, default_scalar_type(D_val))
        h_mesh = fem.Constant(domain, default_scalar_type(1.0 / N))

        # Manufactured solutions
        u_exact = ufl.as_vector([
            ufl.sin(ufl.pi*x[0]) * ufl.cos(ufl.pi*x[1]),
            -ufl.cos(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
        ])

        p_exact = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
        c_exact = ufl.cos(ufl.pi*x[0]) * ufl.cos(ufl.pi*x[1])

        # MMS forcing
        f_stokes = -nu*ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
        s_advdiff = ufl.inner(u_exact, ufl.grad(c_exact)) \
                    - D*ufl.div(ufl.grad(c_exact))

       
        # STOKES SOLVE (saddle-point system)
        
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        a_stokes = (
            nu*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
            - p*ufl.div(v)*ufl.dx
            - q*ufl.div(u)*ufl.dx
        )
        L_stokes = ufl.inner(f_stokes, v)*ufl.dx

        facets = mesh.locate_entities_boundary(
            domain, domain.topology.dim-1,
            lambda x: np.ones(x.shape[1], dtype=bool)
        )

        W0 = W.sub(0)
        Vc, _ = W0.collapse()

        u_bc = fem.Function(Vc)
        u_bc.interpolate(lambda x: np.array([
            np.sin(np.pi*x[0])*np.cos(np.pi*x[1]),
            -np.cos(np.pi*x[0])*np.sin(np.pi*x[1])
        ]))

        dofs_u = fem.locate_dofs_topological(
            (W0, Vc), domain.topology.dim-1, facets
        )
        bc_u = fem.dirichletbc(u_bc, dofs_u, W0)

        # pressure pin
        W1 = W.sub(1)
        Qc, _ = W1.collapse()

        coords = Qc.tabulate_dof_coordinates()
        idx = int(np.argmin(np.linalg.norm(coords, axis=1)))
        xp, yp = coords[idx][:2]

        p_bc_fun = fem.Function(Qc)
        p_bc_fun.x.array[idx] = default_scalar_type(
            np.sin(np.pi*xp)*np.sin(np.pi*yp)
        )

        pin = fem.locate_dofs_geometrical(
            (W1, Qc),
            lambda X: np.isclose(X[0], xp) & np.isclose(X[1], yp)
        )
        bc_p = fem.dirichletbc(p_bc_fun, pin, W1)

        stokes_problem = LinearProblem(
            a_stokes, L_stokes,
            bcs=[bc_u, bc_p],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
        )

        wh = stokes_problem.solve()
        uh = wh.sub(0).collapse()
        ph = wh.sub(1).collapse()

        
        # ADVECTION–DIFFUSION WITH SUPG
        
        c = ufl.TrialFunction(C)
        w = ufl.TestFunction(C)

        eps = fem.Constant(domain, default_scalar_type(1e-10))
        u_norm = ufl.sqrt(ufl.inner(uh, uh) + eps)
        tau = h_mesh/(2.0*u_norm)

        w_supg = tau*ufl.inner(uh, ufl.grad(w))
        Rc = ufl.inner(uh, ufl.grad(c)) - D*ufl.div(ufl.grad(c))

        a_adv = (
            ufl.inner(uh, ufl.grad(c))*w*ufl.dx
            + D*ufl.inner(ufl.grad(c), ufl.grad(w))*ufl.dx
            + w_supg*Rc*ufl.dx
        )

        L_adv = (
            s_advdiff*w*ufl.dx
            + w_supg*s_advdiff*ufl.dx
        )

        c_bc = fem.Function(C)
        c_bc.interpolate(
            lambda x: np.cos(np.pi*x[0])*np.cos(np.pi*x[1])
        )

        dofs_c = fem.locate_dofs_topological(
            C, domain.topology.dim-1, facets
        )
        bc_c = fem.dirichletbc(c_bc, dofs_c)

        adv_problem = LinearProblem(
            a_adv, L_adv,
            bcs=[bc_c],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
        )

        ch = adv_problem.solve()

           # Mesh refinement study used to compute convergence rates
           # in L2 norm for velocity, pressure, and concentration.
        
        err_u = fem.assemble_scalar(
            fem.form(ufl.inner(uh-u_exact, uh-u_exact)*ufl.dx)
        )
        err_p = fem.assemble_scalar(
            fem.form((ph-p_exact)**2*ufl.dx)
        )
        err_c = fem.assemble_scalar(
            fem.form(ufl.inner(ch-c_exact, ch-c_exact)*ufl.dx)
        )

        error_u = np.sqrt(domain.comm.allreduce(err_u, op=MPI.SUM))
        error_p = np.sqrt(domain.comm.allreduce(err_p, op=MPI.SUM))
        error_c = np.sqrt(domain.comm.allreduce(err_c, op=MPI.SUM))

        h = 1.0/N
        h_values.append(h)
        errors_u.append(error_u)
        errors_p.append(error_p)
        errors_c.append(error_c)

        if domain.comm.rank == 0:
            print(f"N={N:3d}, h={h:.4f} | "
                  f"u={error_u:.3e}, p={error_p:.3e}, c={error_c:.3e}")

    return {
        "h": np.array(h_values),
        "u": np.array(errors_u),
        "p": np.array(errors_p),
        "c": np.array(errors_c),
    }


# RATE COMPUTATION


def compute_rates(h, errors):
    rates = []
    for i in range(len(errors)-1):
        rates.append(
            np.log(errors[i]/errors[i+1]) /
            np.log(h[i]/h[i+1])
        )
    return np.array(rates)



# MAIN

if __name__ == "__main__":

    N_values = [8, 16, 32, 64, 128]

    errors = run_convergence_study(N_values)

    if MPI.COMM_WORLD.rank == 0:

        rates_u = compute_rates(errors["h"], errors["u"])
        rates_p = compute_rates(errors["h"], errors["p"])
        rates_c = compute_rates(errors["h"], errors["c"])

        print("\nConvergence rates:")
        print("Velocity:", rates_u)
        print("Pressure:", rates_p)
        print("Concentration:", rates_c)

