import numpy as np
import math
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# TRUSS FUNCTIONS
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def assign_BCs(NL,ENL):
    PD = np.size(NL,1) # Problem Dimension
    NoN = np.size(NL,0) # eNumber of Nodes

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
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# FRAME FUNCTIONS
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def assign_frame_boundary_conditions(ENL, DoP):
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


def get_local_frame_stiffness(element_nodes, element_property, ENL):
    '''
    E = element_property[2] # Youngs Modulus [Pa]
    b = element_property[0] # Width [m]
    h = element_property[1] # Heigth [m]

    A = b*h # Cross Section Area [m^2]
    I = (b*h**3)/12 # Inertia [m^4]
    '''
    A = element_property[0] # Area [m^2]
    I = element_property[1] # Inertia [m^4]
    E = element_property[2] # Youngs Modulus [Pa]

    initial_node = element_nodes[0]
    end_node = element_nodes[1]
    xi = ENL[initial_node - 1, 0] 
    yi = ENL[initial_node - 1, 1]
    xj = ENL[end_node - 1, 0]
    yj = ENL[end_node - 1, 1]

    dx = xj - xi
    dy = yj - yi

    l = math.sqrt((yj - yi)**2 + (xj - xi)**2)  # Length [m]
    # local Stiffness Matrix [k]
    k = np.array([[E*A/l, 0, 0, -E*A/l, 0, 0],
                [0, 12*E*I/(l**3), 6*E*I/(l**2), 0, -12*E*I/(l**3), 6*E*I/(l**2)],
                [0, 6*E*I/(l**2), 4*E*I/l, 0, -6*E*I/(l**2), 2*E*I/l],
                [-E*A/l, 0, 0, E*A/l, 0, 0],
                [0, -12*E*I/(l**3), -6*E*I/(l**2), 0, 12*E*I/(l**3), -6*E*I/(l**2)],
                [0, 6*E*I/(l**2), 2*E*I/l, 0, -6*E*I/(l**2), 4*E*I/l]])
    # Rotation Matrix [Q]
    if dx==0:
        if dy>0:
            beta = math.pi/2
        else:
            beta = -math.pi/2
    else:
        beta = math.atan(dy/dx)

    c = math.cos(beta)
    s = math.sin(beta)
    Q = np.array([[c, s, 0, 0, 0 ,0],
                  [-s, c, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, c, s, 0],
                  [0, 0, 0, -s, c, 0],
                  [0, 0, 0, 0, 0, 1]])
    
    kg = np.transpose(Q)@k@Q

    return kg

def assemble_global_frame_stiffness(ENL, element_list, node_list, node_restraints, element_properties):
    number_elements = element_list.shape[0] # Number of Elements
    nodes_per_element = element_list.shape[1] # Nodes Per Element

    DoP = node_restraints.shape[1] # Problem Dimension
    number_nodes = node_list.shape[0] # Number of Nodes

    K_global = np.zeros([number_nodes*DoP, number_nodes*DoP])
    # Iterating over each element
    for i in range(0, number_elements):
        # Get the initial and end node number
        element_nodes = element_list[i,:]
        # Element Proprerty [bi, hi, Ei]
        element_property = element_properties[i,:]
        # Local Stiffness of the element
        k_element = get_local_frame_stiffness(element_nodes, element_property, ENL)

        for r in range(0, nodes_per_element):
            for p in range(0, DoP):
                for q in range(0,nodes_per_element):
                    for s in range(0,DoP):
                        row = ENL[element_nodes[r] - 1, p + 2 + 2*DoP]
                        column = ENL[element_nodes[q] - 1, s + 2 + 2*DoP]
                        k_contribution = k_element[r*DoP + p, q*DoP + s]
                        K_global[int(row)-1, int(column)-1] = K_global[int(row)-1, int(column)-1] + k_contribution
    return K_global

def  assign_distributed_node_forces(ENL, element_list, element_distributed_loads, DoP):
    node_list = ENL[:,0:2]
    number_elements = element_list.shape[0]
    for i in range(0, number_elements):
        if element_distributed_loads[i] != 0:
            node_ini = element_list[i,0]
            node_end = element_list[i,1]

            xi = node_list[node_ini-1,0]
            yi = node_list[node_ini-1,1]

            xj = node_list[node_end-1,0]
            yj = node_list[node_end-1,1]

            dx = xj - xi
            dy = yj - yi

            # Rotation Matrix [Q]
            if dx==0:
                if dy>0:
                    beta = math.pi/2
                else:
                    beta = -math.pi/2
            else:
                beta = math.atan(dy/dx)

            # Rotation Matrix [Q]
            c = math.cos(beta)
            s = math.sin(beta)
            Q = np.array([[c, s, 0, 0, 0 ,0],
                          [-s, c, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, c, s, 0],
                          [0, 0, 0, -s, c, 0],
                          [0, 0, 0, 0, 0, 1]])


            l = math.sqrt((dx)**2+(dy)**2)

            # Getting the node reactions
            w = element_distributed_loads[i]
            m_ini = (w*l**2)/12
            v_ini = w*l/2

            m_end = -(w*l**2)/12
            v_end = w*l/2

            element_local_forces = -np.array([0, v_ini, m_ini, 0, v_end, m_end])
            # Turning into global coordinates
            # element_global_forces = np.transpose(Q)@element_local_forces@Q
            element_global_forces = np.transpose(Q)@element_local_forces
            print('ELEMENT GLOBAL FORCES',element_global_forces)
            print(element_global_forces.reshape([2,3]))
            element_global_forces = element_global_forces.reshape([2,3]) # Arranging the forces

            # Now we need to actualize the Nodal Forces in the ENL
            # Initial Node
            ENL[node_ini-1, int(2 + 4*DoP):int(2 + 5*DoP)] = ENL[node_ini - 1, int(2 + 4*DoP):int(2 + 5*DoP)] + element_global_forces[0,:]
            # End Node
            ENL[node_end-1, 2 + 4*DoP:2 + 5*DoP] = ENL[node_end - 1, 2 + 4*DoP:2 + 5*DoP] + element_global_forces[1,:]  


    return ENL

def assemble_frame_forces(ENL, node_list, node_restraints):
    DoP = node_restraints.shape[1] # Problem Dimension
    number_nodes = node_list.shape[0] # Number of Nodes
    DoF = 0

    Fp = []
    for i in range(0, number_nodes):
        for j in range(0,DoP):
            if ENL[i, 2+j] == 1:
                DoF += 1
                Fp.append(ENL[i, 2+ 4*DoP + j])
    
    Fp = np.vstack([Fp]).reshape(-1,1)
    return Fp

def assemble_frame_displacements(ENL, node_list, node_restraints):
    DoP = node_restraints.shape[1] # Problem Dimension
    number_nodes = node_list.shape[0] # Number of Nodes
    DoC = 0
    Up = []

    for i in range(0, number_nodes):
        for j in range(0,DoP):
            if ENL[i, 2+j] == -1:
                DoC += 1
                Up.append(ENL[i, 2 + 3*DoP + j])

    Up = np.vstack([Up]).reshape(-1,1)

    return Up

def get_frame_forces_and_displacements(K_global,DoF,DoC,Fp,Up,U_u,Fu):

    U_u = U_u.flatten()
    Fu = Fu.flatten()

    K_UU = K_global[0:DoF, 0:DoF]
    K_UP = K_global[0:DoF, DoF: DoF+DoC]
    K_PU = K_global[DoF:DoF+DoC, 0:DoF]
    K_PP = K_global[DoF:DoF+DoC, DoF:DoF+DoC] 
    F = Fp - np.matmul(K_UP, Up)
    U_u = np.matmul(np.linalg.inv(K_UU), F) # np.linalg.solve() More efficient?
    Fu = np.matmul(K_PU, U_u) + np.matmul(K_PP, Up)

    return (U_u, Fu)

def update_frame_nodes(ENL, U_u, node_list, Fu, node_restraints):
    DoP = node_restraints.shape[1] # Problem Dimension
    number_nodes = node_list.shape[0] # Number of Nodes

    DoF = 0
    DoC = 0

    for i in range(0, number_nodes):
        for j in range(0, DoP):
            if ENL[i, 2 + j] == 1:
                DoF += 1
                ENL[i,2 + 3*DoP + j] = U_u[DoF-1,0]
            else:
                DoC += 1
                ENL[i,2 + 4*DoP + j] = Fu[DoC-1,0]

    return ENL

def get_frame_internal_forces(ENL, element_list, element_properties, element_distributed_loads, DoP):
    number_elements = element_list.shape[0]
    print('-'*75)
    print('Ni, Vi, Mi, Nj, Vj, Mj')
    print('-'*75)
    # Pre-defining the element internal forces array
    element_internal_forces = np.zeros([number_elements, 2*DoP])
    # Pre-defining an element angle array
    element_angles = np.zeros(number_elements)
    # Iterating over each element
    for i in range(number_elements):
        element_property = element_properties[i, :]
        '''
        E = element_property[2] # Youngs Modulus [Pa]
        b = element_property[0] # Width [m]
        h = element_property[1] # Heigth [m]

        A = b*h # Cross Section Area [m^2]
        I = (b*h**3)/12 # Inertia [m^4]
        '''
        A = element_property[0] # Area [m^2]
        I = element_property[1] # Inertia [m^4]
        E = element_property[2] # Youngs Modulus [Pa]

        element_nodes = element_list[i,:]
        node_ini = element_nodes[0]
        node_end = element_nodes[1]

        # Node Coordinates
        xi = ENL[node_ini - 1, 0] 
        yi = ENL[node_ini - 1, 1]
        xj = ENL[node_end - 1, 0]
        yj = ENL[node_end - 1, 1]

        dx = xj - xi
        dy = yj - yi

        # Rotation Matrix [Q]
        if dx==0:
            if dy>0:
                beta = math.pi/2
            else:
                beta = -math.pi/2
        else:
            beta = math.atan(dy/dx)

        # Storing the element angle
        element_angles[i] = beta

        # Element Length
        l = math.sqrt((yj - yi)**2 + (xj - xi)**2)  # Length [m]
        # Rotation Matrix [Q]
        c = math.cos(beta)
        s = math.sin(beta)
        Q = np.array([[c, s, 0, 0, 0 ,0],
                    [-s, c, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, s, 0],
                    [0, 0, 0, -s, c, 0],
                    [0, 0, 0, 0, 0, 1]])

        # local Stiffness Matrix [k]
        k = np.array([[E*A/l, 0, 0, -E*A/l, 0, 0],
                    [0, 12*E*I/(l**3), 6*E*I/(l**2), 0, -12*E*I/(l**3), 6*E*I/(l**2)],
                    [0, 6*E*I/(l**2), 4*E*I/l, 0, -6*E*I/(l**2), 2*E*I/l],
                    [-E*A/l, 0, 0, E*A/l, 0, 0],
                    [0, -12*E*I/(l**3), -6*E*I/(l**2), 0, 12*E*I/(l**3), -6*E*I/(l**2)],
                    [0, 6*E*I/(l**2), 2*E*I/l, 0, -6*E*I/(l**2), 4*E*I/l]])
        
        # Get node global displacements
        node_ini_disp = ENL[node_ini-1, int(2+3*DoP):int(2+4*DoP)]
        node_end_disp = ENL[node_end-1, int(2+3*DoP):int(2+4*DoP)]

        # Assembling element displacements
        element_global_disp = np.zeros(2*DoP) # Pre-defining
        element_global_disp[0:DoP] = node_ini_disp
        element_global_disp[DoP:2*DoP] = node_end_disp

        # Local Displacement
        element_local_disp = Q@element_global_disp

        # Internal Local Forces
        element_internal_force = k@element_local_disp
        # If there is distributed loads we need to actualize the internal forces of the frame
        if element_distributed_loads[i] != 0:
            w = element_distributed_loads[i]
            m_ini = w*(l**2)/12
            m_end = -w*(l**2)/12
            v_ini = w*l/2
            v_end = w*l/2
            element_internal_force = np.array([0, v_ini, m_ini, 0, v_end, m_end]) + element_internal_force

        print('The internal local Forces of the',i+1,'-th element are:')
        print(element_internal_force)

        # Storing the internal forces
        element_internal_forces[i,:] = element_internal_force

    print('-'*75)
    return (element_internal_forces, element_angles)