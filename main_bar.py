import numpy as np
import math
from functions import *
import matplotlib.pyplot as plt

node_list = np.array([[0,0],
                      [0,50],
                      [50,50],
                      [50,0]])
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

# Cross-section Properties
# Element Proprerty [bi, hi, Ei]
element_properties = np.array([[0.25, 0.5, 1e6],
                               [0.25, 0.5, 1e6],
                               [0.25, 0.5, 1e6]])

# Forces
# [Fx, Fy, Mz]
Fu = np.array([[0,0,0],
               [-1,1,5],
               [1,-1,2],
               [0,0,0]])

# Displacement
U_u = np.array([[0,0,0],
                [0,0,0],
                [0,0,0],
                [0,0,0]])

number_nodes = node_list.shape[0]
number_elements = element_list.shape[0]



def assign_boundary_conditions(ENL, DoP):
    number_nodes = ENL.shape[0]
    DoF = 0 # Number Free Degree of Freedom
    DoC = 0 # Number of Fix Degree of Freedom

    for i in range(number_nodes):
        for j in range(DoP):
            if ENL[i, 2 + j] == -1:
                DoC -= 1
                ENL[i, 2 + j + DoP] = DoC
            else:
                DoF += 1
                ENL[i, 2 + j + DoP] = DoF

    for i in range(number_nodes):
        for j in range(DoP):
            if ENL[i, 2 + j + DoP] < 0:
                ENL[i,2 + 2*DoP + j] = abs(ENL[i, 2 + j + DoP]) + DoF
            else:
                ENL[i,2 + 2*DoP + j] = abs(ENL[i, 2 + j + DoP])

    DoC = abs(DoC)
    return (DoF, DoC, ENL)

def get_local_stiffness(element_nodes, element_property, ENL):
    E = element_properties[2] # Youngs Modulus [Pa]
    b = element_properties[0] # Width [m]
    h = element_properties[1] # Heigth [m]

    A = b*h # Cross Section Area [m^2]
    I = b*h**3/12 # Inertia [m^4]
    initial_node = element_nodes[0]
    end_node = element_nodes[1]
    xi = ENL[initial_node - 1, 0] 
    yi = ENL[initial_node - 1, 1]
    xj = ENL[end_node - 1, 0]
    yj = ENL[end_node - 1, 1]

    l = math.sqrt((yj - yi)**2 + (xj - xi)**2)  # Length [m]

    # local Stiffness Matrix [k]
    k = np.array([[E*A/l, 0, 0, -E*A/l, 0, 0],
                [0, 12*E*I/(l**3), 6*E*I/l**2, 0, -12*E*I/(l**3), 6*E*I/(l**2)],
                [0, 6*E*I/(l**2), 4*E*I/l, 0, -6*E*I/(l**2), 2*E*I/l],
                [-E*A/l, 0, 0, E*A/l, 0, 0],
                [0, -12*E*I/(l**3), -6*E*I/(l**2), 0, 12*E*I/(l**3), -6*E*I/(l**2)],
                [0, 6*E*I/(l**2), 2*E*I/l, 0, -6*E*I/(l**2), 4*E*I/l]])
    
    # Rotation Matrix [Q]
    beta = 1
    c = math.cos(beta)
    s = math.sin(beta)
    Q = np.array([[c, s, 0, 0, 0 ,0],
                  [-s, c, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, c, s, 0],
                  [0, 0, 0, c, s, 0],
                  [0, 0, 0, 0, 0, 1]])
    
    kg = np.transpose(Q)@k@Q

    print(k)
    return k

def assemble_global_stiffness(ENL, element_list, node_list, node_restraints, element_properties):
    number_elements = element_list.shape[0] # Number of Elements
    nodes_per_element = element_list.shape[1] # Nodes Per Element

    DoP = node_restraints.shape[1] # Problem Dimension
    number_nodes = node_list.shape[0] # Number of Nodes

    # Iterating over each element
    for i in range(0, number_elements):
        # Get the initial and end node number
        element_nodes = element_list[i,:]
        # Element Proprerty [bi, hi, Ei]
        element_property = element_properties[i,:]
        k_element = get_local_stiffness(element_nodes, element_property, ENL)

# Dimension of the Problem
DoP = node_restraints.shape[1]

# Pre-defining the Extended Node List (ENL)
ENL = np.zeros([number_nodes, 2 + 5*DoP])
# Assign Nodes Coordinates and restraints
ENL[:,0:2] = node_list[:,:]
ENL[:,2:5] = node_restraints[:,:]
print(ENL)
# Assign Boundary Conditions to the ENL
(DoF, DoC, ENL) = assign_boundary_conditions(ENL, DoP)

# Assemble the global stiffness matrix
# K_global = assemble_global_stiffness(ENL, element_list, node_list, element_properties)
print('DoF = ',DoF)
print('DoC = ',DoC)
print(ENL)

'''
for i in range(number_nodes):
    get_local_stiffness(i, ELoE)
'''

plot_pre_process_v1(node_list, node_restraints, Fu, element_list)