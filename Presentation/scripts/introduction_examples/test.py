"""
Very small implicit-Euler simulator for a mass–spring Stanford bunny.
▪ Needs: numpy, scipy, trimesh, matplotlib (optional for preview)
▪ Put a bunny OBJ next to this file, e.g. ‘bun_zipper.ply’ from Stanford.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh                                     # just to load the mesh

# ----------------------------------------------------------------------
# 0.  Input geometry  ---------------------------------------------------
mesh   = trimesh.load("./bunny.obj", process=False)
V0     = mesh.vertices.view(np.ndarray)            # (n,3) rest positions
F      = mesh.faces                                # (m,3) triangles
n      = len(V0)
m_per_vertex = 0.002                               # kg  (tiny bunny)

# ----------------------------------------------------------------------
# 1.  Build a simple edge list and one spring per edge -----------------
edges = np.vstack([tuple(sorted(e)) for tri in F for e in
                   [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]])
rest_len = np.linalg.norm(V0[edges[:,0]] - V0[edges[:,1]], axis=1)
k_spring = 800.0                                   # N / m, same for all

# ----------------------------------------------------------------------
# 2.  Assemble mass (M) and laplacian stiffness (K)  -------------------
row = np.hstack([edges[:,0], edges[:,1], edges[:,0], edges[:,1]])
col = np.hstack([edges[:,0], edges[:,1], edges[:,1], edges[:,0]])
val = np.hstack([ k_spring*np.ones(len(edges)),     # +k on the diagonal blocks
                  k_spring*np.ones(len(edges)),
                 -k_spring*np.ones(len(edges)),    # –k on the off-diagonals
                 -k_spring*np.ones(len(edges))])
K_scalar = sp.coo_matrix((val, (row, col)), shape=(n, n)).tocsr()

M_scalar = sp.diags([m_per_vertex]*n, format="csr")
# Expand to 3×3 block form (x,y,z):
I3       = sp.eye(3, format="csr")
K        = sp.kron(I3, K_scalar)                   # (3n × 3n)
M        = sp.kron(I3, M_scalar)

# ----------------------------------------------------------------------
# 3.  Pre-factorize the constant system matrix with Cholesky -----------
dt = 0.002                                         # s
A  = (M + dt*dt * K).tocsc()                       # SPD
solve = spla.factorized(A)                         # ← Cholesky under the hood

# ----------------------------------------------------------------------
# 4.  Simulation loop ---------------------------------------------------
x  = V0.copy().reshape(-1)                         # flatten to length 3n
v  = np.zeros_like(x)
gravity = np.tile([0, -9.81, 0], n)

for step in range(800):                            # 1.6 s of wobbling
    # explicit forces (gravity only; springs handled implicitly via K)
    f_ext  = gravity * m_per_vertex
    # right-hand side for implicit Euler: (M v  + dt f)
    rhs    = M.dot(v) + dt * f_ext
    v      = solve(rhs)                            # (M + dt² K) v_{t+1} = rhs
    x     += dt * v                                # positions
    # quick-and-dirty floor collision
    below  = (x[1::3] < 0.0)
    x[1::3][below]  = 0.0
    v[1::3][below] *= -0.4

    # optional quick preview every 40 steps
    if step % 40 == 0:
        print(f"frame {step:03d}", end="\r")

# ----------------------------------------------------------------------
# 5.  Dump the deformed mesh -------------------------------------------
mesh.vertices = x.reshape(-1,3)
mesh.export("bunny_deformed.ply")
print("\nWiggly bunny saved → bunny_deformed.ply")