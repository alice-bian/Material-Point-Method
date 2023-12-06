import taichi as ti
import numpy as np
import os

# With arch=ti.gpu, Taichi will first try to run with CUDA. Since CUDA is not supported on a Mac (my machine), Taichi will fall back on Metal.
ti.init(arch=ti.gpu)  # Run on GPU, automatically detect backend

# Initialize particle and grid scalar parameters
quality = 1  # NOTE: Use a larger value to get higher-resolution simulations
n_particles = 150000 * quality ** 2  # number of particles
n_grid = 128 * quality  # number of grid nodes
num_jellos = 3
dx = 0.5 / n_grid
inv_dx = float(n_grid)
dt = 1e-4 / quality
p_vol = (dx * 0.5) ** 2  # particle volume
p_rho = 1  # particle density
p_mass = p_vol * p_rho  # particle mass
E = 2e3  # Young's modulus
nu = 0.2  # Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)) # Lame parameters

# Initialize particle and grid vectors
x = ti.Vector.field(3, dtype=float, shape=n_particles)  # particle positions
v = ti.Vector.field(3, dtype=float, shape=n_particles)  # particle velocities
C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)  # deformation gradient
Jp = ti.field(dtype=float, shape=n_particles)  # particle plastic deformation
grid_v = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(3, dtype=float, shape=())
material = ti.field(dtype=int, shape=n_particles)  # particle material id's

# Taichi function, which can be called by Taichi kernels or other Taichi functions. Returns 3x3 matrix.
@ti.func
def kirchoff_FCR(F, R, J, mu, la):
    return 2 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)  # Compute kirchoff stress for FCR model (remember tau = P F^T)

# Taichi kernel, which can be called from Python-scope to perform computation.
@ti.kernel
def substep():
    # Re-initialize grid quantities
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0

    # Particle state update and scatter to grid (P2G)
    for p in x:
    
        # For particle p, compute base index
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # w is ^N(x), Lectures 10 & 11
        dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]  # derivative of ^N(x) w.r.t. fx

        mu, la = mu_0, lambda_0  # TODO: Opportunity here to modify these to model other materials

        U, sig, V = ti.svd(F[p])
        J = 1.0

        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        
        # Compute Kirchoff Stress
        kirchoff = kirchoff_FCR(F[p], U@V.transpose(), J, mu, la)

        # P2G for velocity and mass AND Force Update!
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # Loop over 3x3x3 grid node neighborhood
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            
            dweight = ti.Vector.zero(float, 3)
            dweight[0] = inv_dx * dw[i][0] * w[j][1] * w[k][2]
            dweight[1] = inv_dx * w[i][0] * dw[j][1] * w[k][2]
            dweight[2] = inv_dx * w[i][0] * w[j][1] * dw[k][2]
            
            force = -p_vol * kirchoff @ dweight

            grid_v[base + offset] += p_mass * weight * (v[p] + C[p] @ dpos)  # momentum transfer
            grid_m[base + offset] += weight * p_mass  # mass transfer

            grid_v[base + offset] += dt * force  # Add force to update velocity, don't divide by mass bc this is actually updating MOMENTUM
    
    # Gravity and Boundary Collision
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:  # No need for epsilon here
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]  # Momentum to velocity
            
            grid_v[i, j, k] += dt * gravity[None] * 30  # gravity
            
            # Wall collisions
            if i < 3 and grid_v[i, j, k][0] < 0:          grid_v[i, j, k][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j, k][0] > 0: grid_v[i, j, k][0] = 0
            if j < 3 and grid_v[i, j, k][1] < 0:          grid_v[i, j, k][1] = 0
            if j > n_grid - 3 and grid_v[i, j, k][1] > 0: grid_v[i, j, k][1] = 0
            if k < 3 and grid_v[i, j, k][2] < 0:          grid_v[i, j, k][2] = 0
            if k > n_grid - 3 and grid_v[i, j, k][2] > 0: grid_v[i, j, k][2] = 0
    
    # Grid to particle (G2P)
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        new_F = ti.Matrix.zero(float, 3, 3)
        
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # Loop over 3x3x3 grid node neighborhood
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]

            dweight = ti.Vector.zero(float,3)
            dweight[0] = inv_dx * dw[i][0] * w[j][1] * w[k][2]
            dweight[1] = inv_dx * w[i][0] * dw[j][1] * w[k][2]
            dweight[2] = inv_dx * w[i][0] * w[j][1] * dw[k][2]

            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            new_F += g_v.outer_product(dweight)
        
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection
        F[p] = (ti.Matrix.identity(float, 3) + (dt * new_F)) @ F[p]  # Update F (explicitMPM way)

@ti.kernel
def reset_jello1():
    group_size = n_particles // num_jellos
    # Jello cube 1
    for i in range(group_size):
        x[i] = [ti.random() * 0.2 + 0.4 + 0.10 * (i // group_size), ti.random() * 0.2 + 0.45 + 0.32 * (i // group_size), ti.random() * 0.2 + 0.7 + 0.22 * (i // group_size)]
        #material[i] = i // group_size  # 0: fluid, 1: jelly, 2: snow
        v[i] = [0, 0, 0]
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 3, 3)

@ti.kernel
def reset_jello2():
    group_size = n_particles // num_jellos
    # Jello cube 2
    for i in range(group_size, group_size * 2):
        x[i] = [ti.random() * 0.2 + 0.4 + 0.10 * (i // (group_size * 2)), ti.random() * 0.2 + 0.75 + 0.32 * (i // (group_size * 2)), ti.random() * 0.2 + 0.8 + 0.22 * (i // (group_size * 2))]
        #material[i] = i // group_size # 0: fluid, 1: jelly, 2: snow
        v[i] = [0, 0, 0]
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 3, 3)

@ti.kernel
def reset_jello3():
    group_size = n_particles // num_jellos
    # Jello cube 3
    for i in range(group_size * 2, n_particles):
        x[i] = [ti.random() * 0.2 + 0.3 + 0.10 * (i // n_particles), ti.random() * 0.2 + 0.25 + 0.32 * (i // n_particles), ti.random() * 0.2 + 0.0 + 0.22 * (i // n_particles)]
        #material[i] = i // group_size  # 0: fluid, 1: jelly, 2: snow
        v[i] = [0, 0, 30]
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 3, 3)

gravity[None] = [0, -9.8, 0]
series_prefix = "render_output/jello.ply"

# Frame 0: First jello cube enters
reset_jello1()
for frame in range(30):
    for s in range(int(2e-3 // dt)):
        substep()
    
    # Generate one .ply file per frame for rendering.
    np_pos = np.reshape(x.to_numpy(), (n_particles, 3))
    dirname = os.path.dirname(series_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    writer = ti.PLYWriter(num_vertices=n_particles)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.export_frame_ascii(frame, series_prefix)

# Frame 30: Second jello cube enters
reset_jello2()
for frame in range(30, 45):
    for s in range(int(2e-3 // dt)):
        substep()
    
    # Generate one .ply file per frame for rendering.
    np_pos = np.reshape(x.to_numpy(), (n_particles, 3))
    dirname = os.path.dirname(series_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    writer = ti.PLYWriter(num_vertices=n_particles)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.export_frame_ascii(frame, series_prefix)

# Frame 45: Third jello cube enters
reset_jello3()
for frame in range(45, 240):
    for s in range(int(2e-3 // dt)):
        substep()
    
    # Generate one .ply file per frame for rendering.
    np_pos = np.reshape(x.to_numpy(), (n_particles, 3))
    dirname = os.path.dirname(series_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    writer = ti.PLYWriter(num_vertices=n_particles)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.export_frame_ascii(frame, series_prefix)
