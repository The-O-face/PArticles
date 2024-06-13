# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 00:18:20 2024

@author: Wolfg
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_particles = 50
dt = 0.01
rest_density = 1000
stiffness = 0.001  # Reduced stiffness to lower repulsion
viscosity = 500
h = 0.1  # Larger smoothing length
mass = 1
gravity = np.array([0, -980])
box_size = 100.0

# Initialize particles
positions = np.random.rand(num_particles, 2) * box_size
velocities = np.zeros((num_particles, 2))
densities = np.zeros(num_particles)
pressures = np.zeros(num_particles)

# SPH Kernel Functions
def poly6_kernel(r, h):
    if 0 <= r < h:
        return (315 / (64 * np.pi * h**9)) * (h**2 - r**2)**3
    return 0

def spiky_gradient_kernel(r, h):
    if 0 <= r < h:
        return -(45 / (np.pi * h**6)) * (h - r)**2
    return 0

def viscosity_laplacian_kernel(r, h):
    if 0 <= r < h:
        return (45 / (np.pi * h**6)) * (h - r)
    return 0

for step in range(1000):
    # Calculate densities
    for i in range(num_particles):
        density = 0
        for j in range(num_particles):
            r = np.linalg.norm(positions[i] - positions[j])
            density += mass * poly6_kernel(r, h)
        densities[i] = density
    
    # Calculate pressures
    pressures = stiffness * (densities - rest_density)
    
    # Calculate forces
    forces = np.zeros_like(velocities)
    
    for i in range(num_particles):
        pressure_force = np.zeros(2)
        viscosity_force = np.zeros(2)
        
        for j in range(num_particles):
            if i != j:
                r_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_vec)
                
                # Pressure force
                if r > 0:
                    pressure_force += -r_vec / r * mass * (pressures[i] + pressures[j]) / (2 * densities[j]) * spiky_gradient_kernel(r, h)
                
                # Viscosity force
                viscosity_force += viscosity * mass * (velocities[j] - velocities[i]) / densities[j] * viscosity_laplacian_kernel(r, h)
        
        forces[i] = pressure_force + viscosity_force + gravity * densities[i]
    
    # Update velocities and positions
    velocities += forces * dt / densities[:, None]
    positions += velocities * dt
    
    # Reflective boundary conditions with damping
    for i in range(num_particles):
        for d in range(2):  # Check both x and y directions
            if positions[i, d] < 0:
                positions[i, d] = 0
                velocities[i, d] *= -0.5  # Dampened reflection
            elif positions[i, d] > box_size:
                positions[i, d] = box_size
                velocities[i, d] *= -0.5  # Dampened reflection
    
    # Visualization
    plt.clf()
    plt.scatter(positions[:, 0], positions[:, 1], color='blue')
    plt.xlim(0, box_size)
    plt.ylim(0, box_size)
    plt.title(f'Step {step}')
    plt.pause(0.01)
    plt.draw()

plt.show()
