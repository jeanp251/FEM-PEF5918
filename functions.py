import numpy as np
import math
import matplotlib.pyplot as plt

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

def plot_pre_process_v0(node_list, node_restraints, Fu, element_list):
    # node_list: np array list of the nodes 
    # node_restraints: np array list of the node constraints [x,y,rz]
    # -1: Fixed
    # +1: Free
    # Fu: Nodal Forces [Fx, Fy, Mz]
    # element_list: np array list of elements of the structure

    number_nodes = node_list.shape[0]
    DoP = node_restraints.shape[1]
    number_elements = element_list.shape[0]

    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(number_nodes):
        x_node = node_list[i,0]
        y_node = node_list[i,1]
        ax.scatter(x_node, y_node, s=50, facecolor='k', edgecolor='k', linewidths = 3, zorder=1)
        ax.annotate(str(i+1), xy=(x_node, y_node), xytext=(x_node + 0.05, y_node + 0.05))

        # PLOT NODAL FORCES
        for j in range(DoP):
            nodal_force = Fu[i,j]
            if nodal_force != 0:
                match j:
                    case 0: # Fx
                        if nodal_force > 0:
                            ax.annotate(str(nodal_force)+'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node - 0.5, y_node + 0.05),
                                        color = 'red')
                            ax.arrow(x_node - 0.5, y_node, 0.45 , 0,
                                    width = 0.04,
                                    color = 'red',
                                    length_includes_head = True)
                        else:
                            ax.annotate(str(abs(nodal_force)) + 'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node + 0.5, y_node + 0.05),
                                        color = 'red')
                            ax.arrow(x_node + 0.5, y_node, -0.45 , 0, 
                                    width = 0.04,
                                    color = 'red',
                                    length_includes_head = True)
                    case 1: # Fy
                        if nodal_force > 0:
                            ax.annotate(str(nodal_force) + 'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node + 0.05, y_node + 0.5),
                                        color = 'red')
                            ax.arrow(x_node, y_node + 0.05, 0, 0.45,
                                    width = 0.04,
                                    color = 'red',
                                    length_includes_head = True)
                        else:
                            ax.annotate(str(abs(nodal_force)) + 'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node + 0.05, y_node + 0.5),
                                        color = 'red')
                            ax.arrow(x_node, y_node + 0.45, 0, -0.45,
                                    width = 0.04,
                                    color = 'red',
                                    length_includes_head = True)
                    case 2: # Mz
                        ax.annotate('Mz='+str(nodal_force)+'N.m',
                                    xy=(x_node, y_node),
                                    xytext=(x_node - 0.5, y_node + 0.5),
                                    color = 'red')
    # PLOT ELEMENTS       
    for i in range(number_elements):
        coord_ini = element_list[i,0]
        coord_end = element_list[i,1]
        # Initial Coordinates
        xi = node_list[coord_ini-1, 0]
        yi = node_list[coord_ini-1, 1]
        # End Coordinates
        xj = node_list[coord_end-1, 0]
        yj = node_list[coord_end-1, 1]
        # Centroid Coordinates
        xg = (xi+xj)/2
        yg = (yi+yj)/2

        x_element = np.array([xi,xj])
        y_element = np.array([yi,yj])

        ax.plot(x_element, y_element, lw=5, zorder=0)
        ax.text(xg, yg, str(i+1), color='blue', bbox=dict(facecolor='white', edgecolor='blue'))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.show()

def plot_pre_process_v1(node_list, node_restraints, Fu, element_list):
    # node_list: np array list of the nodes 
    # node_restraints: np array list of the node constraints [x,y,rz]
    # -1: Fixed
    # +1: Free
    # Fu: Nodal Forces [Fx, Fy, Mz]
    # element_list: np array list of elements of the structure
    #
    number_nodes = node_list.shape[0]
    DoP = node_restraints.shape[1]
    number_elements = element_list.shape[0]
    #
    # PLOT
    # First we need to get a reference dimension to plot all the stuff in proportion to this
    x_min = np.min(node_list[:,0])
    y_min = np.min(node_list[:,1])
    x_max = np.max(node_list[:,0])
    y_max = np.max(node_list[:,1])

    prop_dimension = math.sqrt((x_min-x_max)**2+(y_min-y_max)**2)

    arrow_width = 5.657e-3*prop_dimension # Width of the Force Arrows

    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(number_nodes):
        x_node = node_list[i,0]
        y_node = node_list[i,1]
        ax.scatter(x_node, y_node, s=50, facecolor='k', edgecolor='k', linewidths = 3, zorder=1)
        ax.annotate(str(i+1),
                    xy=(x_node, y_node),
                    xytext=(x_node + 7e-3*prop_dimension, y_node + 7e-3*prop_dimension))

        # PLOT NODAL FORCES
        for j in range(DoP):
            nodal_force = Fu[i,j]
            if nodal_force != 0:
                match j:
                    case 0: # Fx
                        if nodal_force > 0: # Positive Fx
                            ax.annotate(str(nodal_force)+'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node - 7.1e-2*prop_dimension, y_node + 7e-3*prop_dimension),
                                        color = 'red')
                            ax.arrow(x_node - 7.1e-2*prop_dimension, y_node, 6.36e-2*prop_dimension , 0,
                                    width = arrow_width,
                                    color = 'red',
                                    length_includes_head = True)
                        else: # Negative Fx
                            ax.annotate(str(abs(nodal_force)) + 'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node + 7.1e-2*prop_dimension, y_node + 7e-3*prop_dimension),
                                        color = 'red')
                            ax.arrow(x_node + 7.1e-2*prop_dimension, y_node, -6.36e-2*prop_dimension , 0, 
                                    width = arrow_width,
                                    color = 'red',
                                    length_includes_head = True)
                    case 1: # Fy
                        if nodal_force > 0: # Positive Fy
                            ax.annotate(str(nodal_force) + 'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node + 7e-3*prop_dimension, y_node + 7.1e-2*prop_dimension),
                                        color = 'red')
                            ax.arrow(x_node, y_node + 7e-3*prop_dimension, 0, 6.36e-2*prop_dimension,
                                    width = arrow_width,
                                    color = 'red',
                                    length_includes_head = True)
                        else: # Negative Fy
                            ax.annotate(str(abs(nodal_force)) + 'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node + 7e-3*prop_dimension, y_node + 7.1e-2*prop_dimension),
                                        color = 'red')
                            ax.arrow(x_node, y_node + 6.36e-2*prop_dimension, 0, -6.36e-2*prop_dimension,
                                    width = arrow_width,
                                    color = 'red',
                                    length_includes_head = True)
                    case 2: # Mz
                        ax.annotate('Mz='+str(nodal_force)+'N.m',
                                    xy=(x_node, y_node),
                                    xytext=(x_node - 8.5e-2*prop_dimension, y_node + 5e-2*prop_dimension),
                                    color = 'red')
    # PLOT ELEMENTS       
    for i in range(number_elements):
        coord_ini = element_list[i,0]
        coord_end = element_list[i,1]
        # Initial Coordinates
        xi = node_list[coord_ini-1, 0]
        yi = node_list[coord_ini-1, 1]
        # End Coordinates
        xj = node_list[coord_end-1, 0]
        yj = node_list[coord_end-1, 1]
        # Centroid Coordinates
        xg = (xi+xj)/2
        yg = (yi+yj)/2

        x_element = np.array([xi,xj])
        y_element = np.array([yi,yj])

        ax.plot(x_element, y_element, lw=5, zorder=0)
        ax.text(xg, yg, str(i+1), color='blue', bbox=dict(facecolor='white', edgecolor='blue'))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.show()