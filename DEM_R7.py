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
    force: np.ndarray | List[float] = field(default_factory=lambda: [0, 0, 0])
    index: int = 0
    color: str = ""

    def __post_init__(self): #initialized position, velocity and force arrays
        self.position = np.array(self.position, dtype=np.float64)
        self.velocity = np.array(self.velocity, dtype=np.float64)
        self.force = np.array(self.force, dtype=np.float64)

    def __setattr__(self, key, value): #ensure type consistency of acceleration~~~~~~~~~~~
        if key == "acceleration":
            object.__setattr__(self, '_acceleration', np.array(value, dtype = np.float64))
            self.update_velocity_based_on_acceleration()
        else:
            super().__setattr__(key,value)
    
    def __repr__(self) -> str: #~~~~~~~~~~~~~~~~~~~~~~
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

    def apply_force(self, force):
        self.force += np.array(force)

    def reset_force(self):
        self.force = np.zeros(3)

    def check_collision(self, other_sphere): #check sphere collisions based on radius
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

    def update_position(self, container, dt): #update self position based on acceleration and timestep dt
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2
        self.check_wall_collision(container)

    def check_wall_collision(self, container: Container):
        for i in range (3):
            if self.position[i] - self.radius < container.min_bounds[i] or self.position[i] + self.radius > container.max_bounds[i]:
                self.velocity[i] = -self.velocity[i]

#METHODS                
def check_wall_collision_fun(container: Container, sphere: Sphere) -> List[bool]:
    flip = [False, False, False]
    for i in range (3):
        if sphere.position[i] - sphere.radius < container.min_bounds[i] or sphere.position[i] + sphere.radius > container.max_bounds[i]:
            flip[i] = True
    return flip

def update_position_fun(container, sphere, dt) -> np.ndarray: #update sphere position based on acceleration and timestep dt
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
    subspace_occupancy = {}  # A dictionary to keep track of spheres in each subspace
    x_range = np.linspace(container.min_bounds[0], container.max_bounds[0], grid_size)
    y_range = np.linspace(container.min_bounds[1], container.max_bounds[1], grid_size)
    z_range = np.linspace(container.min_bounds[2], container.max_bounds[2], grid_size)

    for sphere in spheres:
        x_idx = np.searchsorted(x_range, sphere.position[0]) - 1
        y_idx = np.searchsorted(y_range, sphere.position[1]) - 1
        z_idx = np.searchsorted(z_range, sphere.position[2]) - 1
        subspace_key = (x_idx, y_idx, z_idx)

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
    
def generate_spheres(num_spheres, force_range, position_range, mass, radius):
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
            #velocity=[0, 0, 0],
            force=force,
            index=index,
            color=color
        )
        sphere.acceleration = sphere.force / mass  # Calculate and set initial acceleration
        spheres.append(sphere)
        
    return spheres
    
def main(spheres, grid_size):
    time_step = 0.1
    total_time = 300
    positions = []
    container = Container(min_bounds=[-40, -40, -40], max_bounds=[40, 40, 40])

    for t in range(int(total_time / time_step)):
        for s in spheres:
            s.reset_force()
            s.update_position(container, time_step)
        subspace_occupancy = update_subspace_occupancy(container, spheres, grid_size)
        for subspace, occupants in subspace_occupancy.items():
            print(f"Time: {t * time_step:.2f}s, Subspace: {subspace}, Spheres: {occupants}")
            
            for idx in occupants:
                sphere = spheres[idx-1]
                print(f"    Sphere {sphere.index} Velocity: {sphere.velocity}, Position: {sphere.position}")

        for i in range(len(spheres)):
            for j in range(i + 1, len(spheres)):
                if spheres[i].check_collision(spheres[j]):
                    spheres[i].collide(spheres[j])
                    print(f"Collision at time {t * time_step:.2f} between {spheres[i].color} Sphere {spheres[i].index} and {spheres[j].color} Sphere {spheres[j].index}")
                    print(f"Positions: {spheres[i].position}, {spheres[j].position}")
                    print(f"Distance between centers: {np.linalg.norm(spheres[i].position - spheres[j].position)}")
                    
        current_positions = []
        for s in spheres:
            current_positions.append(s.position.copy())
        positions.append(current_positions)
        
        #print(f"Time: {t * time_step:2.2e}: {spheres}")

    return np.array(positions), container

#MAIN FUNCTION
if __name__ == "__main__":
    grid_size = 4
    num_spheres = 150
    mass = 1.0
    radius = 1.5
    force_range = [-20, 20]
    position_range = [-30, 30]
    spheres = generate_spheres(num_spheres, force_range, position_range, mass, radius)
    # sphere1 = Sphere(mass=1, radius=1, position=[-20, 0, 0], index=1, color="green")
    # sphere2 = Sphere(mass=1, radius=1, position=[-10, 0, 0], index=2, color="blue")
    # sphere3 = Sphere(mass=1, radius=1, position=[0, 0, 0], index=3, color="orange")
    # sphere4 = Sphere(mass=1, radius=1, position=[10, 0, 0], index = 4, color="purple")
    # sphere5 = Sphere(mass=1, radius=1, position=[20, 0, 0], index = 5, color="red")
    
    # sphere6 = Sphere(mass=1, radius=1, position=[-20, 5, 20], index=6, color="brown")
    # sphere7 = Sphere(mass=1, radius=1, position=[-10, 7, 20], index=7, color="pink")
    # sphere8 = Sphere(mass=1, radius=1, position=[0, -2, 20], index=8, color="gray")
    # sphere9 = Sphere(mass=1, radius=1, position=[10, 0, 20], index = 9, color="olive")
    # sphere10 = Sphere(mass=1, radius=1, position=[20, 0, 20], index = 10, color="cyan")
    
    # sphere11 = Sphere(mass=1, radius=1, position=[0, -25, 0], index=1, color="green")
    # sphere12 = Sphere(mass=1, radius=1, position=[0, -15, 0], index=2, color="blue")
    # sphere13 = Sphere(mass=1, radius=1, position=[0, 5, 0], index=3, color="orange")
    # sphere14 = Sphere(mass=1, radius=1, position=[0, 15, 0], index = 4, color="purple")
    # sphere15 = Sphere(mass=1, radius=1, position=[0, 25, 0], index = 5, color="red")
    
    # sphere16 = Sphere(mass=1, radius=1, position=[0, -25, -20], index=6, color="brown")
    # sphere17 = Sphere(mass=1, radius=1, position=[0, -15, -20], index=7, color="pink")
    # sphere18 = Sphere(mass=1, radius=1, position=[0, 5, -20], index=8, color="gray")
    # sphere19 = Sphere(mass=1, radius=1, position=[0, 15, -20], index = 9, color="olive")
    # sphere20 = Sphere(mass=1, radius=1, position=[0, 25, -20], index = 10, color="cyan")
    
    # force1 = [1, -4, -5]
    # force2 = [-2, 3, 1]
    # force3 = [4, 31, -8]
    # force4 = [25, 16, -6]
    # force5 = [16, 9, 4]
    # force6 = [-7, 8, -3]
    # force7 = [-9, 1, 7]
    # force8 = [8, 5, 10]
    # force9 = [11, -10, -10]
    # force10 = [3, -3, -2]
    
    # sphere1.acceleration = force1
    # sphere2.acceleration = force1
    # sphere3.acceleration = force2
    # sphere4.acceleration = force2
    # sphere5.acceleration = force3
    # sphere6.acceleration = force3
    # sphere7.acceleration = force4
    # sphere8.acceleration = force4
    # sphere9.acceleration = force5
    # sphere10.acceleration = force5
    # sphere11.acceleration = force6
    # sphere12.acceleration = force6
    # sphere13.acceleration = force7
    # sphere14.acceleration = force7
    # sphere15.acceleration = force8
    # sphere16.acceleration = force8
    # sphere17.acceleration = force9
    # sphere18.acceleration = force9
    # sphere19.acceleration = force10
    # sphere20.acceleration = force10
    # spheres = [sphere1, sphere2, sphere3, sphere4, sphere5, sphere6, sphere7, sphere8, sphere9, sphere10, sphere11,
    #            sphere12, sphere13, sphere14, sphere15, sphere16, sphere17, sphere18, sphere19, sphere20]

    positions, container = main(spheres, grid_size)
    positions = np.array(positions)
positions.shape

sizes = [np.pi * (s.radius ** 2) * 21 for s in spheres]
colors = [s.color for s in spheres]

def update_time(frame):
    # Calculate elapsed time in seconds
    elapsed_time = frame * time_step * speed
    # Update the timer text
    timer_text.set_text(f"Time: {elapsed_time:.2f} s")

#configure the plot output
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

#save the animation to a .gif file
writergif = animation.PillowWriter(fps=30)
ani.save("./out/Spheres_R7.gif", writer=writergif)

# Show the figure after saving the animation
plt.show()