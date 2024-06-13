# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 00:30:20 2024

@author: Wolfg
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_particles = 100
dt = 0.001
h = 0.02  # Smoothing length
rest_density = 1000
stiffness = 1000
viscosity = 0.1
gravity = np.array([0, -9.8])
mass = 0.02

# Box boundaries
box_size = 1.0

# Initialize particles
positions = np.random.rand(num_particles, 2) * box_size
velocities = np.zeros((num_particles, 2))
densities = np.zeros(num_particles)
pressures = np.zeros(num_particles)

def poly6_kernel(r, h):
    coeff = 315 / (64 * np.pi * h**9)
    q = (h**2 - r**2)**3
    return coeff * q if 0 <= r < h else 0

def spiky_gradient_kernel(r, h):
    coeff = -45 / (np.pi * h**6)
    q = (h - r)**2
    return coeff * q if 0 <= r < h else 0

def viscosity_laplacian_kernel(r, h):
    coeff = 45 / (np.pi * h**6)
    q = h - r
    return coeff * q if 0 <= r < h else 0

# Simulation loop
for step in range(1000):
    # Calculate densities
    for i in range(num_particles):
        density = 0
        for j in range(num_particles):
            if i != j:
                r = np.linalg.norm(positions[i] - positions[j])
                density += poly6_kernel(r, h)
        densities[i] = density * rest_density
    
    # Calculate pressures
    pressures = stiffness * (densities - rest_density)
    
    # Calculate forces
    forces = np.zeros_like(velocities)
    
    for i in range(num_particles):
        pressure_force = np.zeros(2)
        viscosity_force = np.zeros(2)
        
        for j in range(num_particles):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                
                # Pressure force
                pressure_force += -r_vec / r * mass * (pressures[i] + pressures[j]) / (2 * densities[j]) * spiky_gradient_kernel(r, h)
                
                # Viscosity force
                viscosity_force += viscosity * mass * (velocities[j] - velocities[i]) / densities[j] * viscosity_laplacian_kernel(r, h)
        
        # Gravity force
        gravity_force = gravity * densities[i]
        
        forces[i] = pressure_force + viscosity_force + gravity_force
    
    # Update velocities and positions
    velocities += forces * dt / densities[:, None]
    positions += velocities * dt
    
    # Boundary conditions (simple reflection)
    for i in range(num_particles):
        if positions[i, 0] < 0 or positions[i, 0] > box_size:
            velocities[i, 0] *= -1
        if positions[i, 1] < 0 or positions[i, 1] > box_size:
            velocities[i, 1] *= -1
    
    # Visualization (scatter plot of particles)
    plt.clf()
    plt.scatter(positions[:, 0], positions[:, 1], s=1, color='blue')
    plt.xlim(0, box_size)
    plt.ylim(0, box_size)
    plt.title(f'Step {step}')
    plt.pause(0.001)
    plt.draw()

plt.show()

