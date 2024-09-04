import numpy as np
from functions import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Node List
NL = np.array([[0,0],
              [1,0],
              [0.5,1]])
# Element List
EL = np.array([[1,2],
                [2,3],
                [3,1]])

# Boundary conditions
# -1: Fixed
# 1: Free
DorN = np.array([[-1, -1],
                 [1, -1],
                 [1, 1]])
# Forces
# Initially zero
Fu = np.array([[0,0],
               [0,0],
               [0,-20]])
# Displacement
U_u = np.array([[0,0],
                [0,0],
                [0,0]])

E = 10**6 # Young's Modulus
A = 0.01 # Cross Section

# Problem Dimension
PD = np.size(NL, 1)
# Number of Nodes
NoN = np.size(NL,0)
# Extended Node List
ENL = np.zeros([NoN, 6*PD])

ENL[:,0:PD] = NL[:,:]

ENL[:, PD:2*PD] = DorN[:,:]

print(NL)
print(EL)
print('Problem Dimension',PD)
print('Number of Nodes',NoN)
print('EXTENDED Node List')
print(ENL)

(ENL, DOFs, DOCs) = assign_BCs(NL, ENL)
print('EXTENDED Node List')
print(ENL)

K = assemble_stiffness(ENL, EL, NL, E, A)

ENL[:,4*PD:5*PD] = U_u[:,:]
ENL[:,5*PD:6*PD] = Fu[:,:]

U_u = U_u.flatten()
Fu = Fu.flatten()

Fp = assemble_forces(ENL, NL) 
Up = assemble_displacements(ENL, NL)

print(Fp)
print(Up)

K_UU = K[0:DOFs, 0:DOFs]
K_UP = K[0:DOFs, DOFs: DOFs + DOCs]
K_PU = K[DOFs:DOFs + DOCs, 0:DOFs ]
K_PP = K[DOFs:DOFs + DOCs, DOFs:DOFs + DOCs ] 

F = Fp - np.matmul(K_UP, Up)
U_u = np.matmul(np.linalg.inv(K_UU), F)
Fu = np.matmul(K_PU, U_u) + np.matmul(K_PP, Up)

print('Displacements U_u')
print(U_u)
print('Forces Fu')
print(Fu)

ENL = update_nodes(ENL, U_u, NL, Fu)
print('Final ENL')
np.set_printoptions(suppress=True, precision=4)
print(ENL)

# ----------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------

scale = 100 # Exaggeration Scale
coordinates = []
dispx_array = []

for i in range(NoN):
    dispx = ENL[i,PD*4]
    dispy = ENL[i,PD*4 + 1]

    x = ENL[i,0] + dispx*scale
    y = ENL[i,1] + dispy*scale

    dispx_array.append(dispx)
    coordinates.append(np.array([x,y]))

coordinates = np.vstack(coordinates)
dispx_array = np.vstack(dispx_array)

x_scatter = []
y_scatter = []
color_x = []

for i in range(0, np.size(EL,0)):
    x1 = coordinates[EL[i,0] -1, 0]
    x2 = coordinates[EL[i,1] -1, 0]
    y1 = coordinates[EL[i,0] -1, 1]
    y2 = coordinates[EL[i,1] -1, 1]

    dispx_EL = np.array([dispx_array[EL[i,0] - 1], dispx_array[EL[i,1] - 1]])

    if x1 == x2:
        x = np.linspace(x1,x2,200)
        y = np.linspace(y1,y2,200)
    else:
        m = (y2-y1)/(x2-x1)
        x = np.linspace(x1,x2,200)
        y = m*(x-x1) + y1

    x_scatter.append(x)
    y_scatter.append(y)

    color_x.append(np.linspace(np.abs(dispx_EL[0]), np.abs(dispx_EL[1]), 200))

x_scatter = np.vstack([x_scatter]).flatten()
y_scatter = np.vstack([y_scatter]).flatten()
color_x = np.vstack([color_x]).flatten()


dispFigure = plt.figure()
ax_dispx = dispFigure.add_subplot()

color_map = plt.get_cmap('jet')
ax_dispx.scatter(x_scatter, y_scatter, c = color_x, cmap = color_map, s=10, edgecolor = 'none')

norm_x = Normalize(np.abs(dispx_array.min()), np.abs(dispx_array.max()))
dispFigure.colorbar(ScalarMappable(norm = norm_x, cmap = color_map))

plt.show()