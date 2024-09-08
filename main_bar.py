import numpy as np
import math
from functions import *
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------
# INPUT DATA
# ----------------------------------------------------------------------------
node_list = np.array([[0,0],
                      [0,5],
                      [5,5],
                      [5,0]])
# -1: Fixed
# +1: Free
# [x,y,rz]
node_restraints = np.array([[-1, -1, -1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [-1, -1, -1]])

element_list = np.array([[1,2],
                         [2,3],
                         [3,4]])
# ----------------------------------------------------------------------------
# CROSS-SECTION PROPERTIES
# ----------------------------------------------------------------------------
# Element Proprerty [bi, hi, Ei]
element_properties = np.array([[0.25, 0.5, 1e6],
                               [0.25, 0.5, 1e6],
                               [0.25, 0.5, 1e6]])
# ----------------------------------------------------------------------------
# FORCES
# ----------------------------------------------------------------------------
# [Fx, Fy, Mz]
Fu = np.array([[0,0,0],
               [1,0,0],
               [0,0,0],
               [0,0,0]])
# Displacement
# Initially zeros
U_u = np.array([[0,0,0],
                [0,0,0],
                [0,0,0],
                [0,0,0]])

number_nodes = node_list.shape[0]
number_elements = element_list.shape[0]

# Dimension of the Problem
DoP = node_restraints.shape[1]
# Pre-defining the Extended Node List (ENL)
ENL = np.zeros([number_nodes, 2 + 5*DoP])
# Assign Nodes Coordinates and restraints
ENL[:,0:2] = node_list[:,:]
ENL[:,2:5] = node_restraints[:,:]

# Assign Boundary Conditions to the ENL
(DoF, DoC, ENL) = assign_bar_boundary_conditions(ENL, DoP)

print('DoF = ',DoF)
print('DoC = ',DoC)

# Assemble the global stiffness matrix
K_global = assemble_global_bar_stiffness(ENL, element_list, node_list, node_restraints, element_properties)

print(K_global)
# Assing Displacements and Forces to the ENL
ENL[:,2+3*DoP:2+4*DoP] = U_u[:,:]
ENL[:,2+4*DoP:2+5*DoP] = Fu[:,:]

U_u = U_u.flatten()
Fu = Fu.flatten()
Fp = assemble_bar_forces(ENL, node_list, node_restraints)
Up = assemble_bar_displacements(ENL, node_list, node_restraints)

K_UU = K_global[0:DoF, 0:DoF]
K_UP = K_global[0:DoF, DoF: DoF+DoC]
K_PU = K_global[DoF:DoF+DoC, 0:DoF]
K_PP = K_global[DoF:DoF+DoC, DoF:DoF+DoC] 

F = Fp - np.matmul(K_UP, Up)
U_u = np.matmul(np.linalg.inv(K_UU), F)
Fu = np.matmul(K_PU, U_u) + np.matmul(K_PP, Up)

print('Displacements U_u')
print(U_u)
print(U_u.shape)
print('Forces Fu')
print(Fu)
print(Fu.shape)

ENL = update_bar_nodes(ENL, U_u, node_list, Fu, node_restraints)   

print('Final ENL')
np.set_printoptions(suppress=True, precision=4)
print(ENL)
# plot_bar_pre_process_v1(node_list, node_restraints, Fu, element_list)