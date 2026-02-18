

# Diffusion coefficient study (Case D1)
# Investigates transport behaviour for baseline diffusion.




from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from pathlib import Path

default_scalar_type = np.float64

# MESH LOADINGs
# Read DFG 2D benchmark mesh with marked boundaries:
# - Tag 1: Inlet boundary
# - Tag 2: Wall boundaries (top and bottom)
# - Tag 3: Circular obstacle
# - Tag 4: Outlet boundary


comm = MPI.COMM_WORLD

# Read mesh on rank 0 only (avoids MPI read/communication issues)

domain, cell_tags, facet_tags = gmshio.read_from_msh(
        "dfg_benchmark_2d.msh",
        MPI.COMM_SELF,
        gdim=2
    )

# FUNCTION SPACES SETUP
degree = 1  # Base polynomial degree (P1)

# Velocity space (u):
# Quadratic (P2) vector Lagrange elements.
# Higher-order velocity improves accuracy of gradients
# and satisfies the Taylor–Hood stability requirement.
V = fem.functionspace(domain, ("Lagrange", degree + 1, (2,)))

# Pressure space (p):
# Linear (P1) scalar Lagrange elements.
# Paired with P2 velocity (Taylor–Hood element) to ensure inf-sup stability.
Q = fem.functionspace(domain, ("Lagrange", degree))

# Concentration space (c):
# Linear (P1) scalar Lagrange elements.
# Sufficient for scalar transport and consistent with pressure space.
C = fem.functionspace(domain, ("Lagrange", degree))

if domain.comm.rank == 0:
    print("=" * 80)
    print("Coupled Stokes-Advection-Diffusion Problem")
    print("=" * 80)

# =============================================================================
# STEP 1: SOLVE STOKES PROBLEM
# =============================================================================
# Solve for velocity u and pressure p from Stokes equations:
#   -ν∇²u + ∇p = 0  (momentum equation)
#   ∇·u = 0         (incompressibility constraint)

if domain.comm.rank == 0:
    print("\n1. Solving Stokes problem...")

# Create mixed element for Taylor-Hood P2-P1 formulation
# This ensures stable velocity-pressure coupling
gdim = domain.geometry.dim
P2 = element("Lagrange", domain.basix_cell(), degree + 1, shape=(gdim,))
P1 = element("Lagrange", domain.basix_cell(), degree)
TH = mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

# Define trial and test functions for mixed formulation
# u: velocity (trial), p: pressure (trial)
# v: velocity (test), q: pressure (test)
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# Physical parameters
# Kinematic viscosity (ν = 1e-3 gives low Reynolds number, Stokes regime)
nu = fem.Constant(domain, default_scalar_type(1e-3))

# Body force (zero in this benchmark problem)
f = fem.Constant(domain, default_scalar_type((0.0, 0.0)))

# Weak formulation of Stokes equations
# Find (u,p) such that for all test functions (v,q):
# First term: ν∫(∇u:∇v)dx - viscous diffusion
# Second term: -∫p(∇·v)dx - pressure gradient on momentum
# Third term: -∫q(∇·u)dx - incompressibility constraint
a_stokes = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a_stokes += -p * ufl.div(v) * ufl.dx - q * ufl.div(u) * ufl.dx
L_stokes = ufl.inner(f, v) * ufl.dx


# BOUNDARY CONDITIONS FOR STOKES PROBLEM


def inflow_velocity(x):
    
# Inlet boundary condition (DFG benchmark)
# -------------------------------------------------
# Parabolic (Poiseuille) velocity profile:
#
#   u_x(y) = 4 * Umax * y * (H - y) / H^2
#   u_y(y) = 0
#
# - Maximum velocity at channel centerline (y = H/2)
# - Zero velocity at walls (y = 0 and y = H)
# - Purely horizontal inflow
#
# This provides a physically consistent laminar inflow
# for the incompressible Stokes problem.

    U = 0.3  # Maximum inlet velocity
    return np.stack([
        4.0 * U * x[1] * (0.41 - x[1]) / 0.41**2,  # u_x component
        np.zeros_like(x[1])                         # u_y = 0 (horizontal flow)
    ])

# -------------------------------------------------
# Boundary conditions for Stokes problem
# -------------------------------------------------

# Extract velocity subspace from mixed space
W0 = W.sub(0)                      # velocity component
V_collapsed, _ = W0.collapse()     # standalone velocity space

# -------------------------
# Inlet: prescribed inflow
# -------------------------
# Parabolic velocity profile (Poiseuille flow)
u_in = fem.Function(V_collapsed)
u_in.interpolate(inflow_velocity)

inlet_dofs = fem.locate_dofs_topological(
    (W0, V_collapsed), 1, facet_tags.find(1)
)
bc_in = fem.dirichletbc(u_in, inlet_dofs, W0)

# -------------------------
# Walls: no-slip condition
# -------------------------
# Velocity is zero on top and bottom walls
gdim = domain.geometry.dim
u_wall = fem.Function(V_collapsed)
u_wall.interpolate(lambda x: np.zeros((gdim, x.shape[1])))

wall_dofs = fem.locate_dofs_topological(
    (W0, V_collapsed), 1, facet_tags.find(2)
)
bc_wall = fem.dirichletbc(u_wall, wall_dofs, W0)

# -------------------------
# Obstacle: no-slip
# -------------------------
obstacle_dofs = fem.locate_dofs_topological(
    (W0, V_collapsed), 1, facet_tags.find(3)
)
bc_obstacle = fem.dirichletbc(u_wall, obstacle_dofs, W0)

# -------------------------
# Outlet: natural boundary condition
# -------------------------
# No Dirichlet condition imposed.
# Zero traction condition is automatically enforced
# by the weak formulation.

bcs = [bc_in, bc_wall, bc_obstacle]

# SOLVE STOKES SYSTEM

# Use direct solver (MUMPS) for robustness
# MUMPS: Multifrontal Massively Parallel Sparse direct solver
problem = LinearProblem(
    a_stokes,
    L_stokes,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",              # No Krylov iteration (direct solve)
        "pc_type": "lu",                     # LU factorization
        "pc_factor_mat_solver_type": "mumps" # Use MUMPS for parallel LU
    }
)

# Solve the linear system
wh = problem.solve()

# Extract velocity and pressure from mixed solution
uh = wh.sub(0).collapse()  # Velocity field
ph = wh.sub(1).collapse()  # Pressure field
uh.name = "Velocity"
ph.name = "Pressure"

if domain.comm.rank == 0:
    print("   Stokes problem solved successfully")


# STEP 2: SOLVE ADVECTION-DIFFUSION PROBLEM

# Solve for concentration c using computed velocity field u:
#   u·∇c - D∇²c = 0
# where u is known from Step 1 (one-way coupling)

if domain.comm.rank == 0:
    print("\n2. Solving advection-diffusion problem...")

# Define trial and test functions for concentration
c = ufl.TrialFunction(C)  # Concentration (trial)
w = ufl.TestFunction(C)   # Concentration (test)

# Diffusion coefficient
# D = 5e-4 gives Péclet number Pe ~ O(100)
D = fem.Constant(domain, default_scalar_type(1e-3))

# Weak formulation of advection-diffusion equation
# First term: ∫(u·∇c)w dx - advection by fluid flow
# Second term: D∫(∇c·∇w)dx - diffusive spreading
# Note: Advection term is LINEAR in c since u is fixed from Step 1
a_advdiff = ufl.inner(uh, ufl.grad(c)) * w * ufl.dx + D * ufl.inner(
    ufl.grad(c), ufl.grad(w)
) * ufl.dx
L_advdiff = fem.Constant(domain, default_scalar_type(0.0)) * w * ufl.dx

# BOUNDARY CONDITIONS FOR CONCENTRATION


def inflow_concentration(x):
    """
    Concentrated inlet condition: c = 1 in narrow band around y = 0.2
    
    This creates a tracer that gets advected by the flow and diffuses.
    The narrow band (±0.05) allows observation of dispersion.
    """
    return np.where(np.abs(x[1] - 0.2) <= 0.05, 1.0, 0.0)

# Inlet boundary condition: prescribed concentration band
c_in = fem.Function(C)
c_in.interpolate(inflow_concentration)
inlet_dofs_c = fem.locate_dofs_topological(C, 1, facet_tags.find(1))
bc_c = fem.dirichletbc(c_in, inlet_dofs_c)

# Natural boundary conditions (automatically enforced):
# - Walls/obstacle: c(u·n) - D∂c/∂n = 0 → -D∂c/∂n = 0 (since u = 0 no-slip)
#   This means zero diffusive flux (insulating walls)
# - Outlet: -D∂c/∂n = 0 (zero diffusive flux, advection carries c out)
bcs_advdiff = [bc_c]


# SOLVE ADVECTION-DIFFUSION SYSTEM

advdiff_problem = LinearProblem(
    a_advdiff,
    L_advdiff,
    bcs=bcs_advdiff,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
)

ch = advdiff_problem.solve()
ch.name = "Concentration"

if domain.comm.rank == 0:
    print("   Advection-diffusion problem solved successfully")


# EXPORT RESULTS FOR VISUALIZATION

# Save solution fields in VTX format (optimized for ParaView)
# VTX format: ADIOS2-based, efficient for large parallel datasets

results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)


# Export concentration field
filename = results_folder / "concentration_D_1e-3"
with io.VTXWriter(domain.comm, str(filename.with_suffix(".bp")), [ch]) as vtx:
    vtx.write(0.0)


# SOLUTION SUMMARY

if domain.comm.rank == 0:
    print("\n3. Results exported to results/ directory")
    print("   - stokes_velocity.bp")
    print("   - stokes_pressure.bp")
    print("   - concentration_D_1e-3.bp")
    print("\n" + "=" * 80)
    print("Visualization:")
    print("  paraview results/stokes_velocity.bp")
    print("=" * 80)   


