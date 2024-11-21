import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib import colors as mcolors
from dataclasses import field, dataclass
from typing import List

#OBJECT CLASSES
@dataclass
class Container:
    min_bounds: np.ndarray
    max_bounds: np.ndarray
    
    def __post_init__(self):
        self.min_bounds = np.array(self.min_bounds, dtype=np.float64)
        self.max_bounds = np.array(self.max_bounds, dtype=np.float64)
        
@dataclass
class Sphere:
    mass: float
    radius: float
    position: np.ndarray | List[float] = field(default_factory=lambda: [0, 0, 0])
    velocity: np.ndarray | List[float] = field(default_factory=lambda: [0, 0, 0])
    _acceleration: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=np.float64))
    _gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81], dtype=np.float64))
    force: np.ndarray | List[float] = field(default_factory=lambda: [0, 0, 0])
    index: int = 0
    color: str = ""
    shear_modulus: float = 0.0

    def __post_init__(self): #initialized position, velocity and force arrays
        self.position = np.array(self.position, dtype=np.float64)
        self.velocity = np.array(self.velocity, dtype=np.float64)
        self.force = np.array(self.force, dtype=np.float64)
        self.volume = 4/3 *np.pi * ( self.radius**3)
        self.density = self.mass / self.volume

    def __setattr__(self, key, value): #ensure type consistency of acceleration~~~~~~~~~~~
        if key == "acceleration":
            object.__setattr__(self, '_acceleration', np.array(value, dtype = np.float64))
            self.update_velocity_based_on_acceleration()
        else:
            super().__setattr__(key,value)
    
    # def __setattr__(self, key, value):
    #     if key == "gravity":
    #         object.__setattr__(self, '_gravity', np.array(value, dtype = np.float64))
    #         self.update_velocity_based_on_gravity()
    #     else:
    #         super().__setattr__(key,value)
            
    def __setattr__(self, key, value):
        if key == "gravity":
            object.__setattr__(self, '_gravity', True)
            self.update_velocity_based_on_gravity()
        else:
            super().__setattr__(key,value)
    
    def __repr__(self) -> str:
        return self.info
            
    @property
    def info(self) -> str:
        fmt_str = (f"Sphere{self.index:02d}(position={self.position}, "
                f"velocity={self.velocity}, ")
        return fmt_str
        
    def acceleration(self) -> np.ndarray:
        return self._acceleration
        
    def update_velocity_based_on_acceleration(self):
        self.velocity += self._acceleration
        
    # def update_velocity_based_on_gravity(self):
    #     self.velocity += self._gravity
        
    # def update_velocity_based_on_gravity(self):
    #     if gravity == True:
    #         self.velocity += self._gravity
    #     else:
    #         self.velocity += self._acceleration

    # def apply_force(self, force):
    #     self.force += np.array(force)

    def reset_force(self):
        self.force = np.zeros(3)

    def check_collision(self, other_sphere):
        distance = np.linalg.norm(self.position - other_sphere.position)
        return distance <= self.radius + other_sphere.radius

    def collide(self, other_sphere):
        m1, m2 = self.mass, other_sphere.mass
        v1, v2 = self.velocity, other_sphere.velocity
        p1, p2 = self.position, other_sphere.position

        normal = (p2 - p1) / np.linalg.norm(p2 - p1)
        overlap = self.radius + other_sphere.radius - np.linalg.norm(p2 - p1)
        self.position -= overlap * normal * (m2 / (m1 + m2))
        other_sphere.position += overlap * normal * (m1 / (m1 + m2))

        new_v1 = v1 - 2 * m2 / (m1 + m2) * np.dot(v1 - v2, p1 - p2) / np.linalg.norm(p1 - p2)**2 * (p1 - p2)
        new_v2 = v2 - 2 * m1 / (m1 + m2) * np.dot(v2 - v1, p2 - p1) / np.linalg.norm(p2 - p1)**2 * (p2 - p1)
        self.velocity, other_sphere.velocity = new_v1, new_v2

    def update_position(self, container, dt):
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2
        self.check_wall_collision(container)

    def check_wall_collision(self, container: Container):
        for i in range (3):
            if self.position[i] - self.radius < container.min_bounds[i] or self.position[i] + self.radius > container.max_bounds[i]:
                self.velocity[i] = -self.velocity[i]
    def check_wall_collision(self, container: Container):
        for i in range(3):
            if self.position[i] - self.radius < container.min_bounds[i]:
                self.velocity[i] = -self.velocity[i]
                self.position[i] = container.min_bounds[i] + self.radius
            elif self.position[i] + self.radius > container.max_bounds[i]:
                self.velocity[i] = -self.velocity[i]
                self.position[i] = container.max_bounds[i] - self.radius
                
    def penalty_force(self, other_sphere, time_step):
        distance = np.linalg.norm(self.position - other_sphere.position)
        overlap = self.radius + other_sphere.radius - distance
        
        if overlap > 0 and time_step > 0:
            average_shear_modulus = (self.shear_modulus + other_sphere.shear_modulus)/2
            penalty_force_magnitude = average_shear_modulus * overlap
            
            direction = (other_sphere. position - self.position) / distance
            
            self.force = ((self.density*other_sphere.density* (self.radius**3)*(other_sphere.radius**3))/(self.density*(self.radius**3)+other_sphere.density*(other_sphere.radius**3))*(overlap/time_step))
            other_sphere.force = (((self.shear_modulus*other_sphere.shear_modulus)/(self.shear_modulus + other_sphere.shear_modulus))*np.sqrt((self.radius*other_sphere.radius)/(self.radius + other_sphere.radius))*(overlap**(3/2)))
            
#METHODS                
def check_wall_collision_fun(container: Container, sphere: Sphere) -> List[bool]:
    flip = [False, False, False]
    for i in range (3):
        if sphere.position[i] - sphere.radius < container.min_bounds[i] or sphere.position[i] + sphere.radius > container.max_bounds[i]:
            flip[i] = True
    return flip

def update_position_fun(container, sphere, dt) -> np.ndarray:
    new_position = sphere.position.copy()
    new_velocity = sphere.velocity.copy()
    acceleration = sphere.force / sphere.mass
    new_velocity += acceleration * dt
    new_position += sphere.velocity * dt + 0.5 * acceleration * dt ** 2
    collided = check_wall_collision_fun(container, sphere)
    for i in range(3):
        if collided[i]:
            new_velocity[i] *= -1
    return new_position, new_velocity

def update_subspace_occupancy(container: Container, spheres, grid_size):
    #empty subpspace dictionary
    subspace_occupancy = {}
    #subdivision of container space by determined bound size, with grid size as user input
    x_range = np.linspace(container.min_bounds[0], container.max_bounds[0], grid_size)
    y_range = np.linspace(container.min_bounds[1], container.max_bounds[1], grid_size)
    z_range = np.linspace(container.min_bounds[2], container.max_bounds[2], grid_size)

    for sphere in spheres:
        #organize sphere position into corrected index tuple for checking occupancy in subspaces
        x_idx = np.searchsorted(x_range, sphere.position[0]) - 1
        y_idx = np.searchsorted(y_range, sphere.position[1]) - 1
        z_idx = np.searchsorted(z_range, sphere.position[2]) - 1
        subspace_key = (x_idx, y_idx, z_idx)
        
        #sorts each sphere into its appropriate subspace
        if subspace_key not in subspace_occupancy:
            subspace_occupancy[subspace_key] = []
        subspace_occupancy[subspace_key].append(sphere.index)
    return subspace_occupancy

def draw_grid_lines(ax, container, grid_size):
    x_lines = np.linspace(container.min_bounds[0], container.max_bounds[0], grid_size)
    y_lines = np.linspace(container.min_bounds[1], container.max_bounds[1], grid_size)
    z_lines = np.linspace(container.min_bounds[2], container.max_bounds[2], grid_size)

    # for x in x_lines:
    #     for y in y_lines:
    #         ax.plot([x, x], [y, y], zs=[container.min_bounds[2], container.max_bounds[2]], color='red', linestyle='--', alpha=0.5)
    # for z in z_lines:
    #     for x in x_lines:
    #         ax.plot([x, container.max_bounds[0]], [container.min_bounds[1], container.min_bounds[1]], zs=z, color='red', linestyle='--', alpha=0.5)
    #     for y in y_lines:
    #         ax.plot([container.min_bounds[0], container.min_bounds[0]], [y, container.max_bounds[1]], zs=z, color='red', linestyle='--', alpha=0.5)

    for x in x_lines:
        ax.plot([x, x], [container.min_bounds[1], container.max_bounds[1]], zs=container.min_bounds[2], color='red')
        ax.plot([x, x], [container.min_bounds[1], container.max_bounds[1]], zs=container.max_bounds[2], color='red')

    for y in y_lines:
        ax.plot([container.min_bounds[0], container.max_bounds[0]], [y, y], zs=container.min_bounds[2], color='red')
        ax.plot([container.min_bounds[0], container.max_bounds[0]], [y, y], zs=container.max_bounds[2], color='red')

    for z in z_lines:
        ax.plot([container.min_bounds[0], container.max_bounds[0]], [container.min_bounds[1], container.min_bounds[1]], zs=z, color='red')
        ax.plot([container.min_bounds[0], container.max_bounds[0]], [container.max_bounds[1], container.max_bounds[1]], zs=z, color='red')

def customize_axes(ax):
    ax.set_axis_off()
    
# !-----------------------
# FOR RANDOM SPHERE GENERATION
def generate_spheres(num_spheres, force_range, position_range, mass, radius, shear_modulus=1.0):
    spheres =[]
    sphere_colors = list(mcolors.CSS4_COLORS.values())

    for i in range(num_spheres):
        index = i + 1
        color = random.choice(sphere_colors)
        force = np.random.uniform(force_range[0], force_range[1], 3)
        position = np.random.uniform(position_range[0], position_range[1], 3)
        sphere = Sphere(
            mass=mass,
            radius=radius,
            position=position,
            force=force,
            index=index,
            color=color,
            shear_modulus=shear_modulus
        )
        sphere.acceleration = sphere.force / mass
        spheres.append(sphere)
        
    return spheres
# !-----------------------
    
def main(spheres, grid_size):
    gravity = False
    time_step = 0.1
    total_time = 300
    positions = []
    container = Container(min_bounds=[-50, -50, -50], max_bounds=[50, 50, 50])

    for t in range(int(total_time / time_step)):
        for s in spheres:
            s.reset_force()
            s.update_position(container, time_step)
        subspace_occupancy = update_subspace_occupancy(container, spheres, grid_size)
        for subspace, occupants in subspace_occupancy.items():
            print(f"Time: {t * time_step:.2f}s, Subspace: {subspace}, Spheres: {occupants}, # Spheres: {len(occupants)}")
            
            for idx in occupants:
                sphere = spheres[idx-1]
                velocity_formatted = [f"{component:.2f}" for component in sphere.velocity]
                position_formatted = [f"{component:.2f}" for component in sphere.position]
                print(f"    Sphere {sphere.index} - Velocity: {velocity_formatted}, Position: {position_formatted}")
            
            for i in range(len(occupants)):
                for j in range(i + 1, len(occupants)):
                    sphere1 = spheres[occupants[i]-1]
                    sphere2 = spheres[occupants[j]-1]

                    distance = np.linalg.norm(sphere1.position - sphere2.position)
                    if distance < 3:
                        print(f"    Distance between Sphere {sphere1.index} and Sphere {sphere2.index}: {distance - sphere1.radius - sphere2.radius}")

        for i in range(len(spheres)):
            for j in range(i + 1, len(spheres)):
                sphere1 = spheres[i]
                sphere2 = spheres[j]
                if sphere1.check_collision(sphere2):
                    sphere1.penalty_force(sphere2, time_step)
        
                if spheres[i].check_collision(spheres[j]):
                    spheres[i].collide(spheres[j])
                    print(f"Collision at time {t * time_step:.2f} between {spheres[i].color} Sphere {spheres[i].index} and {spheres[j].color} Sphere {spheres[j].index}")
                    print(f"Positions: {spheres[i].position}, {spheres[j].position}")
                    print(f"Distance between centers: {np.linalg.norm(spheres[i].position - spheres[j].position)}")
                    
        current_positions = []
        for s in spheres:
            current_positions.append(s.position.copy())
        positions.append(current_positions)

    return np.array(positions), container

#MAIN FUNCTION
if __name__ == "__main__":
    grid_size = 4
    
    # !---------------------
    # FOR RANDOM SPHERE GENERATION
    num_spheres = 600
    mass = 1.0
    radius = 2
    force_range = [-20, 20]
    position_range = [-40, 40]
    spheres = generate_spheres(num_spheres, force_range, position_range, mass, radius)
    # !---------------------

    sphere1 = Sphere(mass=1, radius=1, position=[-8, 0, 0], index=1, color = "red")
    sphere2 = Sphere(mass=1, radius=3, position=[7, 0, 0], index=2, color = "blue")
    sphere3 = Sphere(mass=1, radius=1, position=[10, 0, 0], index=3, color = "green")
    
    # gravity = [5, 0, 0]
    # sphere1.acceleration = gravity
    # sphere1.shear_modulus = 1.0
    # sphere2.shear_modulus = 1.0
    # sphere3.shear_modulus = 1.0
    # spheres = [sphere1, sphere2, sphere3]

    positions, container = main(spheres, grid_size)
    positions = np.array(positions)

#PLOTTING
sizes = [np.pi * (s.radius ** 2) * 21 for s in spheres]
colors = [s.color for s in spheres]

def update_time(frame):
    elapsed_time = frame * time_step * speed
    timer_text.set_text(f"Time: {elapsed_time:.2f} s")

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], c=colors, s=sizes, alpha=1)

ax.set_xlim([container.min_bounds[0], container.max_bounds[0]])
ax.set_ylim([container.min_bounds[1], container.max_bounds[1]])
ax.set_zlim([container.min_bounds[2], container.max_bounds[2]])

timer_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

draw_grid_lines(ax, container, grid_size)

customize_axes(ax)

def update_ax(frame):
    scatter._offsets3d = (positions[frame, :, 0], positions[frame, :, 1], positions[frame, :, 2])
    update_time(frame)

speed = 5
time_step = .01
ani = animation.FuncAnimation(
    fig, update_ax, frames=math.ceil(positions.shape[0] / speed),
    interval=50, blit=False, repeat=True
)

writergif = animation.PillowWriter(fps=30)
ani.save("Spheres_R9.gif", writer=writergif)
plt.show()

## where this code left off: I was attempting to separate acceleration into gravity (in z direction)
## and applied forces, so that gravity can be turned on or off as a boolean