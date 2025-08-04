"""
Interactive mass–spring Stanford bunny simulation with mouse control.
▪ Needs: numpy, scipy, trimesh, matplotlib
▪ Click and drag to apply forces to the bunny!
▪ Press 'r' to reset, 'space' to pause/unpause
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time

class InteractiveBunny:
    def __init__(self):
        # Load mesh
        self.mesh = trimesh.load("./bunny.obj", process=False)
        self.V0 = self.mesh.vertices.view(np.ndarray).copy()  # (n,3) rest positions
        self.F = self.mesh.faces                               # (m,3) triangles
        self.n = len(self.V0)
        self.m_per_vertex = 0.002                             # kg
        
        # Center and scale the bunny for better visualization
        center = np.mean(self.V0, axis=0)
        self.V0 -= center
        self.V0 *= 10  # Make it bigger
        self.V0[:, 1] += 1  # Lift it up a bit
        
        # Simulation parameters
        self.dt = 0.005  # Smaller timestep for stability
        self.gravity = np.tile([0, -9.81, 0], self.n)
        self.damping = 0.99  # Velocity damping
        
        # Build springs
        self.setup_springs()
        
        # Current state
        self.x = self.V0.copy().reshape(-1)  # Current positions (flattened)
        self.v = np.zeros_like(self.x)       # Current velocities
        
        # Mouse interaction
        self.mouse_force = np.zeros_like(self.x)
        self.drag_vertex = None
        self.drag_start = None
        self.is_dragging = False
        
        # Animation control
        self.paused = False
        self.reset_requested = False
        
        # Setup visualization
        self.setup_plot()
        
    def setup_springs(self):
        """Build spring system"""
        # Create edges from triangles
        edges = [tuple(sorted(e)) for tri in self.F for e in
                [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]]
        edges = np.array(list(set(edges)))  # Remove duplicates
        
        rest_len = np.linalg.norm(self.V0[edges[:,0]] - self.V0[edges[:,1]], axis=1)
        k_spring = 500.0  # Reduced stiffness for more interactive feel
        
        # Build stiffness matrix
        row = np.hstack([edges[:,0], edges[:,1], edges[:,0], edges[:,1]])
        col = np.hstack([edges[:,0], edges[:,1], edges[:,1], edges[:,0]])
        val = np.hstack([k_spring*np.ones(len(edges)),
                        k_spring*np.ones(len(edges)),
                        -k_spring*np.ones(len(edges)),
                        -k_spring*np.ones(len(edges))])
        K_scalar = sp.coo_matrix((val, (row, col)), shape=(self.n, self.n)).tocsr()
        
        M_scalar = sp.diags([self.m_per_vertex]*self.n, format="csr")
        
        # Expand to 3D
        I3 = sp.eye(3, format="csr")
        self.K = sp.kron(I3, K_scalar)
        self.M = sp.kron(I3, M_scalar)
        
        # Pre-factorize system matrix
        A = (self.M + self.dt*self.dt * self.K).tocsc()
        self.solve = spla.factorized(A)
        
    def setup_plot(self):
        """Setup matplotlib 3D plot"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initial plot
        vertices = self.x.reshape(-1, 3)
        self.scatter = self.ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                                     c=vertices[:, 1], cmap='viridis', s=1, alpha=0.8)
        
        # Plot some edges for structure
        edges_subset = self.get_surface_edges()[:200]  # Just some edges for visualization
        for edge in edges_subset:
            pts = vertices[[edge[0], edge[1]]]
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'k-', alpha=0.1, linewidth=0.5)
        
        # Set plot properties
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-2, 4)
        self.ax.set_zlim(-3, 3)
        self.ax.set_title("Interactive Bunny - Click and drag to apply forces!\nPress 'r' to reset, 'space' to pause")
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def get_surface_edges(self):
        """Get edges for visualization"""
        edges = []
        for tri in self.F:
            edges.extend([(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])])
        return list(set([tuple(sorted(e)) for e in edges]))
        
    def find_closest_vertex(self, screen_pos):
        """Find the closest vertex to mouse click"""
        if screen_pos is None:
            return None
            
        vertices = self.x.reshape(-1, 3)
        
        # Project 3D points to 2D screen coordinates
        screen_coords = []
        for i, vertex in enumerate(vertices):
            # This is a simplified projection - matplotlib's actual projection is more complex
            proj = self.ax.transData.transform([vertex[0], vertex[1], vertex[2]])
            if len(proj) >= 2:
                screen_coords.append((proj[0], proj[1], i))
        
        if not screen_coords:
            return None
            
        # Find closest vertex
        min_dist = float('inf')
        closest_vertex = None
        
        for sx, sy, vertex_idx in screen_coords:
            dist = np.sqrt((sx - screen_pos[0])**2 + (sy - screen_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_vertex = vertex_idx
                
        return closest_vertex if min_dist < 50 else None  # Within 50 pixels
        
    def on_mouse_press(self, event):
        """Handle mouse press"""
        if event.inaxes != self.ax:
            return
            
        self.drag_vertex = self.find_closest_vertex((event.x, event.y))
        if self.drag_vertex is not None:
            self.is_dragging = True
            self.drag_start = (event.x, event.y)
            
    def on_mouse_release(self, event):
        """Handle mouse release"""
        self.is_dragging = False
        self.drag_vertex = None
        self.mouse_force.fill(0)  # Clear forces
        
    def on_mouse_move(self, event):
        """Handle mouse movement"""
        if not self.is_dragging or self.drag_vertex is None or event.inaxes != self.ax:
            return
            
        # Calculate force based on mouse movement
        if self.drag_start is not None:
            dx = (event.x - self.drag_start[0]) * 0.01
            dy = (event.y - self.drag_start[1]) * 0.01
            
            # Apply force to the selected vertex and nearby vertices
            force_magnitude = 100.0
            vertices = self.x.reshape(-1, 3)
            
            # Clear previous forces
            self.mouse_force.fill(0)
            
            # Apply force to selected vertex
            self.mouse_force[self.drag_vertex*3:self.drag_vertex*3+3] = [dx*force_magnitude, dy*force_magnitude, 0]
            
            # Apply smaller forces to nearby vertices
            selected_pos = vertices[self.drag_vertex]
            for i, vertex in enumerate(vertices):
                if i != self.drag_vertex:
                    dist = np.linalg.norm(vertex - selected_pos)
                    if dist < 0.3:  # Within 0.3 units
                        falloff = np.exp(-dist * 5)  # Exponential falloff
                        self.mouse_force[i*3:i*3+3] += [dx*force_magnitude*falloff*0.3, 
                                                       dy*force_magnitude*falloff*0.3, 0]
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == ' ':  # Space to pause/unpause
            self.paused = not self.paused
            print("Paused" if self.paused else "Unpaused")
        elif event.key == 'r':  # R to reset
            self.reset_requested = True
            print("Resetting...")
            
    def reset_simulation(self):
        """Reset to initial state"""
        self.x = self.V0.copy().reshape(-1)
        self.v = np.zeros_like(self.x)
        self.mouse_force.fill(0)
        self.reset_requested = False
        
    def physics_step(self):
        """Perform one physics simulation step"""
        if self.paused:
            return
            
        # External forces (gravity + mouse forces)
        f_ext = self.gravity * self.m_per_vertex + self.mouse_force
        
        # Implicit Euler step
        rhs = self.M.dot(self.v) + self.dt * f_ext
        self.v = self.solve(rhs)
        
        # Apply damping
        self.v *= self.damping
        
        # Update positions
        self.x += self.dt * self.v
        
        # Floor collision
        below = (self.x[1::3] < -2.0)
        self.x[1::3][below] = -2.0
        self.v[1::3][below] *= -0.6
        
    def update_plot(self, frame):
        """Animation update function"""
        if self.reset_requested:
            self.reset_simulation()
            
        # Perform physics step
        self.physics_step()
        
        # Update visualization
        vertices = self.x.reshape(-1, 3)
        self.scatter._offsets3d = (vertices[:, 0], vertices[:, 1], vertices[:, 2])
        
        # Update colors based on height
        colors = vertices[:, 1]
        self.scatter.set_array(colors)
        
        return [self.scatter]
        
    def start_animation(self):
        """Start the interactive animation"""
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=20, blit=False)
        plt.show()

if __name__ == "__main__":
    print("Loading interactive bunny...")
    bunny = InteractiveBunny()
    print("Starting animation - click and drag to interact!")
    bunny.start_animation() 