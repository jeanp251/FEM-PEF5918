import numpy as np
import math
import matplotlib.pyplot as plt

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

        ax.plot(x_element, y_element, lw=2, color = 'black', zorder=0)
        ax.text(xg, yg, str(i+1), color='blue', bbox=dict(facecolor='white', edgecolor='blue'))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.show()

def plot_frame_pre_process_v1(node_list, node_restraints, Fu, element_list, element_distributed_loads):
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
            node_restraint = node_restraints[i,:]
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
        # PLOT NODE RESTRAINTS
        if np.any(node_restraint<0):
            delta_x = 3.5355e-2*prop_dimension
            delta_y = delta_x
            delta_x_text = delta_x
            delta_y_text = 6.3639e-2*prop_dimension
            restraint_x = np.array([x_node, x_node+delta_x, x_node-delta_x, x_node])
            restraint_y = np.array([y_node, y_node-delta_y, y_node-delta_y, y_node])
            ax.plot(restraint_x, restraint_y, lw = 2, color = 'red')
            
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
    # Iterating over each element
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

        ax.plot(x_element, y_element, lw=2, color = 'blue', zorder=0)
        ax.text(xg, yg, str(i+1), color='blue', bbox=dict(facecolor='white', edgecolor='blue'))

        # PLOT ELEMENT DISTRIBUTED LOADS
        if element_distributed_loads[i] != 0:
            w = element_distributed_loads[i]
            dx = xj - xi
            dy = yj - yi
            if dx==0:
                if dy>0:
                    beta = math.pi/2
                else:
                    beta = -math.pi/2
            else:
                beta = math.atan(dy/dx)

            # Plotting Diagram
            xwi = xi - w*math.sin(beta)
            ywi = yi + w*math.cos(beta)

            xwj = xj - w*math.sin(beta)
            ywj = yj + w*math.cos(beta)

            x_distributed_moment = np.array([xi, xj, xwj, xwi, xi])
            y_distributed_moment = np.array([yi, yj, ywj, ywi, yi])

            ax.plot(x_distributed_moment, y_distributed_moment, lw=2, color = 'green', zorder=1)

            # Plotting Text
            x_text = (xwi+xwj)/2
            y_text = ywi
            w_text = str(np.round(w, decimals = 2))

            ax.annotate(w_text + 'N/m',
                        xy=(x_text, y_text),
                        xytext=(x_text, y_text), 
                        color = 'green',
                        zorder = 2)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.show()