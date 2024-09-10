import numpy as np
import math
from functions import *
from input_frame import *
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------
# INPUT DATA
# ----------------------------------------------------------------------------
(node_list, node_restraints, element_list, element_properties, Fu, U_u) = get_frame_input_data()

number_nodes = node_list.shape[0]
number_elements = element_list.shape[0]

# Dimension of the Problem
DoP = node_restraints.shape[1]
# Pre-defining the Extended Node List (ENL)
ENL = np.zeros([number_nodes, 2 + 5*DoP])

# PLOT STRUCTURE
plot_frame_pre_process_v0(node_list, node_restraints, Fu, element_list)

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
post_process_frame  (ENL,DoP, scale_factor=500)

node_displacements = ENL[:,int(2+3*DoP):int(2+4*DoP)]
print('node displacements')
print(node_displacements)
internal_forces = ENL[:, 2+4*DoP:2+5*DoP]
print('internal forces')
print(internal_forces)