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
# BAR FUNCTIONS
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def plot_frame_pre_process_v0(node_list, node_restraints, Fu, element_list):
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
    # PLOT NODES
    for i in range(number_nodes):
        x_node = node_list[i,0]
        y_node = node_list[i,1]
        # PLOT NODES INFO
        ax.scatter(x_node, y_node, s=50, facecolor='k', edgecolor='k', linewidths = 3, zorder=1)
        ax.annotate(str(i+1), xy=(x_node, y_node), xytext=(x_node + 0.05, y_node + 0.05))

        for j in range(DoP):
            nodal_force = Fu[i,j]
            node_restraint = node_restraints[i,:]
            # PLOT NODAL FORCES
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
            # PLOT NODE RESTRAINTS
        if np.any(node_restraint<0):
            delta_x = 0.25
            delta_y = 0.25
            delta_x_text = 0.25
            delta_y_text = 0.45
            restraint_x = np.array([x_node, x_node+delta_x, x_node-delta_x, x_node])
            restraint_y = np.array([y_node, y_node-delta_y, y_node-delta_y, y_node])
            ax.plot(restraint_x, restraint_y, lw = 2)
            
            if node_restraint[0]<0:
                ux = 'Ux'
            else:
                ux = ''

            if node_restraint[1]<0:
                uy = 'Uy'
            else:
                uy = ''

            if node_restraint[2]<0:
                uz = 'Rz'
            else:
                uz = ''
            restraint_info = ux+uy+uz

            ax.text(x_node-delta_x_text,
                    y_node-delta_y_text,
                    restraint_info,
                    color='red',
                    bbox=dict(facecolor='white', edgecolor='red'))

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

def plot_frame_pre_process_v1(node_list, node_restraints, Fu, element_list):
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
    # PLOT NODES
    for i in range(number_nodes):
        x_node = node_list[i,0]
        y_node = node_list[i,1]
        # PLOT NODE INFO
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
    E = element_property[2] # Youngs Modulus [Pa]
    b = element_property[0] # Width [m]
    h = element_property[1] # Heigth [m]

    A = b*h # Cross Section Area [m^2]
    I = (b*h**3)/12 # Inertia [m^4]
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
    # np.linalg.solve()
    F = Fp - np.matmul(K_UP, Up)
    U_u = np.matmul(np.linalg.inv(K_UU), F)
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

def post_process_frame(ENL, DoP, scale_factor):
    # Gettting the structure info from the ENL
    node_list = ENL[:,0:2]
    number_nodes = node_list.shape[0]
    node_displacements = ENL[:,int(2+3*DoP):int(2+4*DoP)]
    node_restraints = ENL[:, 2:int(2+DoP)]
    node_forces = ENL[:, 2+4*DoP:2+5*DoP]
    # Nodes coordinates
    x_ini = node_list[:,0]
    y_ini = node_list[:,1]
    x_end = x_ini + node_displacements[:,0]*scale_factor
    y_end = y_ini + node_displacements[:,1]*scale_factor

    fig, ax = plt.subplots(figsize=(10,10))
    # Plotting Node Displacements
    ax.scatter(x_ini, y_ini, s=50, facecolor='k', edgecolor='k', linewidths = 1, zorder=1)
    ax.scatter(x_end, y_end, s=50, facecolor='k', edgecolor='k', linewidths = 1, zorder=1)
    # Plotting undeformed structure
    ax.plot(x_ini, y_ini, dashes = [4,4], color = 'green')
    ax.plot(x_end, y_end, color = 'red')

    # Plotting Node Reactions
    for i in range(0,number_nodes):
        boundary_node_conditions = node_restraints[i,:]
        # Evaluating if any of the Node has restraints
        if np.any(boundary_node_conditions<0):
            x_node = node_list[i,0]
            y_node = node_list[i,1]
            reactions = np.round(node_forces[i,:], decimals=2)
            for j in range(0,DoP):
                reaction = reactions[j]
                match j:
                    case 0: # Fx
                        if reaction > 0:
                            ax.annotate(str(reaction)+'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node - 0.5, y_node + 0.05),
                                        color = 'red')
                            ax.arrow(x_node - 0.5, y_node, 0.45 , 0,
                                    width = 0.04,
                                    color = 'red',
                                    length_includes_head = True)
                        else:
                            ax.annotate(str(abs(reaction))+'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node - 0.5, y_node + 0.05),
                                        color = 'red')
                            ax.arrow(x_node - 0.05, y_node, -0.45 , 0,
                                    width = 0.04,
                                    color = 'red',
                                    length_includes_head = True)
                    case 1: # Fy
                        if reaction > 0:
                            ax.annotate(str(reaction) + 'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node + 0.05, y_node - 0.5),
                                        color = 'red')
                            ax.arrow(x_node, y_node - 0.5, 0, 0.45,
                                    width = 0.04,
                                    color = 'red',
                                    length_includes_head = True)
                        else:
                            ax.annotate(str(abs(reaction)) + 'N',
                                        xy=(x_node, y_node),
                                        xytext=(x_node + 0.05, y_node - 0.5),
                                        color = 'red')
                            ax.arrow(x_node, y_node - 0.05, 0, -0.45,
                                    width = 0.04,
                                    color = 'red',
                                    length_includes_head = True)

                    case 2: # Mz
                        ax.annotate('Mz='+str(reaction)+'N.m',
                                    xy=(x_node, y_node),
                                    xytext=(x_node - 0.5, y_node + 0.25),
                                    color = 'red')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.show()