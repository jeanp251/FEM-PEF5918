import numpy as np
import math
from input_frame import *
from frame_pre_process import *
from functions import *
from frame_pos_process import *
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------
# INPUT DATA
# ----------------------------------------------------------------------------
problem = 'P1v2'

(node_list, node_restraints, element_list, element_properties, Fu, element_distributed_loads, U_u) = get_frame_input_data(problem)

number_nodes = node_list.shape[0]
number_elements = element_list.shape[0]

# Dimension of the Problem
DoP = node_restraints.shape[1]
# Pre-defining the Extended Node List (ENL)
ENL = np.zeros([number_nodes, 2 + 5*DoP])

# PLOT STRUCTURE
plot_frame_pre_process_v1(node_list, node_restraints, Fu, element_list, element_distributed_loads)

# Assign Nodes Coordinates and restraints
ENL[:,0:2] = node_list[:,:]
ENL[:,2:5] = node_restraints[:,:]

# Assign Boundary Conditions to the ENL
(DoF, DoC, ENL) = assign_frame_boundary_conditions(ENL, DoP)

print('DoF = ',DoF)
print('DoC = ',DoC)

# Assemble the global stiffness matrix
K_global = assemble_global_frame_stiffness(ENL, element_list, node_list, node_restraints, element_properties)

print(K_global)

# Assing Displacements and Forces to the ENL
ENL[:,2+3*DoP:2+4*DoP] = U_u[:,:]
ENL[:,2+4*DoP:2+5*DoP] = Fu[:,:]

# Assing the node concentrate loads due to the distributed loads
# to the ENL
ENL = assign_distributed_node_forces(ENL, element_list, element_distributed_loads, DoP)

Fp = assemble_frame_forces(ENL, node_list, node_restraints)
Up = assemble_frame_displacements(ENL, node_list, node_restraints)

(U_u, Fu) = get_frame_forces_and_displacements(K_global, DoF, DoC, Fp, Up, U_u, Fu)

print('Displacements U_u')
print(U_u)
print(U_u.shape)
print('Forces Fu')
print(Fu)
print(Fu.shape)

ENL = update_frame_nodes(ENL, U_u, node_list, Fu, node_restraints)   

print('Final ENL')
np.set_printoptions(suppress=True, precision=4)
print(ENL)
# POS-PROCESS
plot_deformation_frame(ENL,DoP, scale_factor=1)

node_displacements = ENL[:,int(2+3*DoP):int(2+4*DoP)]
print('node displacements')
print(node_displacements)
node_reactions = ENL[:, 2+4*DoP:2+5*DoP]
print('node reactions')
print(node_reactions)

(element_internal_forces, element_angles) = get_frame_internal_forces(ENL, element_list, element_properties, element_distributed_loads, DoP)

print(element_internal_forces)
print(element_angles)

plot_frame_moment_diagram(ENL, element_list, element_internal_forces, element_angles, scale_factor=1)
plot_frame_shear_diagram(ENL, element_list, element_internal_forces, element_angles, scale_factor=1)
print(ENL[:,14:17]) # REMOVE!!!