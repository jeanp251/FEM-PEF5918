import numpy as np
import math
def assign_BCs(NL,ENL):
    PD = np.size(NL,1) # Problem Dimension
    NoN = np.size(NL,0) # Number of Nodes

    DOFs = 0
    DOCs = 0

    for i in range(0,NoN):
        for j in range(0,PD):
            if ENL[i, PD + j] == -1:
                DOCs -= 1
                ENL[i,2*PD+j] = DOCs
            else:
                DOFs += 1
                ENL[i, 2*PD+j] = DOFs

    for i in range(0,NoN):
        for j in range(0,PD):
            if ENL[i, 2*PD + j] < 0:
                ENL[i, 3*PD+j] = abs(ENL[i,2*PD+j]) + DOFs
            else:
                ENL[i, 3*PD+j] = abs(ENL[i,2*PD+j])


    DOCs = abs(DOCs)

    return (ENL, DOFs, DOCs)

def assemble_stiffness(ENL, EL, NL, E, A):
    NoE = np.size(EL,0) # Number of Elements
    NPE = np.size(EL,1) # Nodes Per Element

    PD = np.size(NL,1) # Problem Dimension
    NoN = np.size(NL,0) # Number of Nodes

    K = np.zeros([NoN*PD, NoN*PD])
    # Iterating over each element
    for i in range(0,NoE):
        n1 = EL[i, 0:NPE] # Get the initial and end node number
        k = element_stiffness(n1, ENL, E, A)
        
        for r in range(0,NPE):
            for p in range(0, PD):
                for q in range(0,NPE):
                    for s in range(0,PD):
                        row = ENL[n1[r]-1, p+3*PD]
                        column = ENL[n1[q]-1, s+3*PD]
                        value = k[r*PD + p, q*PD + s]
                        K[int(row)-1, int(column)-1] = K[int(row)-1, int(column)-1] + value
    return K

def element_stiffness(n1,ENL,E,A):
    x1 = ENL[n1[0]-1, 0]
    y1 = ENL[n1[0]-1, 1]

    x2 = ENL[n1[1]-1, 0]
    y2 = ENL[n1[1]-1, 1]
    # Length of the element
    L = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    C = (x2-x1)/L
    S = (y2-y1)/L

    k = (E*A)/L*np.array([[C**2, C*S, -C**2, -C*S],
                          [C*S, S**2, -C*S, -S**2],
                          [-C**2, -C*S, C**2, C*S],
                          [-C*S, -S**2, C*S, S**2]])
    return k


def assemble_forces(ENL,NL):
    PD = np.size(NL,1) # Problem Dimension
    NoN = np.size(NL,0) # Number of Nodes
    DOF = 0

    Fp = []

    for i in range(0, NoN):
        for j in range(0,PD):
            if ENL[i, PD+j] == 1:
                DOF += 1
                Fp.append(ENL[i, 5*PD+j])

    Fp = np.vstack([Fp]).reshape(-1,1)

    return Fp


def assemble_displacements(ENL,NL):
    PD = np.size(NL,1) # Problem Dimension
    NoN = np.size(NL,0) # Number of Nodes
    DOC = 0

    Up = []

    for i in range(0, NoN):
        for j in range(0,PD):
            if ENL[i, PD+j] == -1:
                DOC += 1
                Up.append(ENL[i, 4*PD+j])

    Up = np.vstack([Up]).reshape(-1,1)

    return Up

def update_nodes(ENL, U_u, NL, Fu):
    PD = np.size(NL,1) # Problem Dimension
    NoN = np.size(NL,0) # Number of Nodes

    DOFs = 0
    DOCs = 0

    for i in range(0,NoN):
        for j in range(0,PD):
            if ENL[i, PD + j] == 1:
                DOFs += 1
                ENL[i,4*PD+j] = U_u[DOFs-1]
            else:
                DOCs += 1
                ENL[i, 5*PD+j] = Fu[DOCs-1]

    return ENL