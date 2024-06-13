# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 00:14:31 2024

@author: Wolfg
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_particles_blue = 500
num_particles_yellow = 500
num_particles = num_particles_blue + num_particles_yellow
dt = 0.01  # Time step
viscosity = 0.5  # Increased viscosity for more damping
gravity = np.array([0, -98])  # Stronger gravity for more observable interactions

# Lennard-Jones parameters for blue particles
epsilon_blue = 0.5
sigma_blue = 0.0001  # Smaller sigma for repulsion range
# Lennard-Jones parameters for yellow particles
epsilon_yellow = 1.0  # Stronger attraction among yellow particles
sigma_yellow = 0.0001  # Smaller sigma for repulsion range
cutoff = 2.5 * max(sigma_blue, sigma_yellow)
box_size = 2.0  # Size of the simulation box

# Initialize particles
positions = np.random.rand(num_particles, 2) * box_size  # Random positions
velocities = np.zeros((num_particles, 2))

# Assign colors to particles (0 for blue, 1 for yellow)
colors = np.zeros(num_particles, dtype=int)
colors[-num_particles_yellow:] = 1

# Simulation loop
def lennard_jones_force(r, epsilon, sigma):
    if r < cutoff:
        return 24 * epsilon * ((2 * (sigma / r)**13) - ((sigma / r)**7)) / r
    else:
        return 0

for step in range(1000):
    # Calculate forces
    forces = np.zeros_like(velocities)
    
    # Apply gravity
    forces += gravity
    
    # Apply viscosity
    forces -= viscosity * velocities
    
    # Apply Lennard-Jones force
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            delta_r = positions[j] - positions[i]
            dist = np.linalg.norm(delta_r)
            if dist > 0.01 * sigma_blue:  # Avoid singularity at very small distances
                if colors[i] == 1 and colors[j] == 1:
                    # Yellow-yellow interaction
                    force_magnitude = lennard_jones_force(dist, epsilon_yellow, sigma_yellow)
                else:
                    # Blue-blue or blue-yellow interaction
                    force_magnitude = lennard_jones_force(dist, epsilon_blue, sigma_blue)
                force_vector = force_magnitude * delta_r / dist
                forces[i] -= force_vector
                forces[j] += force_vector
    
    # Update velocities and positions
    velocities += forces * dt
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
    for i in range(num_particles):
        if colors[i] == 0:
            plt.scatter(positions[i, 0], positions[i, 1], color='blue')
        else:
            plt.scatter(positions[i, 0], positions[i, 1], color='yellow')
    plt.xlim(0, box_size)
    plt.ylim(0, box_size)
    plt.title(f'Step {step}')
    plt.pause(0.01)
    plt.draw()

plt.show()
