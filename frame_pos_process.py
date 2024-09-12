import numpy as np
import math
import matplotlib.pyplot as plt

def plot_deformation_frame(ENL, DoP, scale_factor):
    node_list = ENL[:,0:2]

    # First we need to get a reference dimension to plot all the stuff in proportion to this
    x_min = np.min(node_list[:,0])
    y_min = np.min(node_list[:,1])
    x_max = np.max(node_list[:,0])
    y_max = np.max(node_list[:,1])

    prop_dimension = math.sqrt((x_min-x_max)**2+(y_min-y_max)**2) # Porportional Dimension

    # Gettting the structure info from the ENL
    number_nodes = node_list.shape[0]
    node_displacements = ENL[:,int(2+3*DoP):int(2+4*DoP)]

    node_displacement_max = np.max(node_displacements[:,0:2])
    node_displacements_scaled = (1/node_displacement_max)*(prop_dimension/7.0711)*scale_factor*node_displacements

    node_restraints = ENL[:, 2:int(2+DoP)]
    node_forces = ENL[:, 2+4*DoP:2+5*DoP]
    # Nodes coordinates
    x_ini = node_list[:,0]
    y_ini = node_list[:,1]
    # x_end = x_ini + node_displacements[:,0]*scale_factor
    # y_end = y_ini + node_displacements[:,1]*scale_factor

    x_end = x_ini + node_displacements_scaled[:,0]
    y_end = y_ini + node_displacements_scaled[:,1]

    fig, ax = plt.subplots(figsize=(10,10))
    # Plotting Node Displacements
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

def plot_frame_moment_diagram(ENL, element_list, element_internal_forces, element_angles, scale_factor):

    number_elements = element_list.shape[0]

    element_moments = np.zeros([number_elements, 2])
    element_moments[:,0] = element_internal_forces[:,2]
    element_moments[:,1] = element_internal_forces[:,5]

    moment_max = element_moments.max()
    element_scaled_moments = (1/moment_max)*scale_factor*element_moments

    fig, ax = plt.subplots(figsize=(10,10))

    for i in range(0,number_elements):
        # Getting the nodes of the element
        element_nodes = element_list[i,:]
        node_ini = element_nodes[0]
        node_end = element_nodes[1]

        # Node Coordinates
        xi = ENL[node_ini - 1, 0]
        yi = ENL[node_ini - 1, 1]
        xj = ENL[node_end - 1, 0]
        yj = ENL[node_end - 1, 1]

        # Element angle
        beta = element_angles[i]

        # Now we are plotting the scaled moments
        moment_ini = element_scaled_moments[i,0]
        moment_end = -element_scaled_moments[i,1] # Negative for plot purposes

        # Initial moment Coordinates
        xmi = xi - moment_ini*math.sin(beta)
        ymi = yi + moment_ini*math.cos(beta)

        # Final Moment Coordinates
        xmj = xj - moment_end*math.sin(beta)
        ymj = yj + moment_end*math.cos(beta)

        # Text Initial Coordinates
        xti = (xi+xmi)/2
        yti = (yi+ymi)/2
        # Text End Coordinates
        xtj = (xj+xmj)/2
        ytj = (yj+ymj)/2
    
        moment_ini_value = str(np.round(element_moments[i,0], decimals=2))
        moment_end_value = str(np.round(element_moments[i,1], decimals=2))

        # Setting the vector of the moment diagram
        x_moment = np.array([xi, xmi, xmj, xj])
        y_moment = np.array([yi, ymi, ymj, yj])

        # Setting the vecotrs of the element
        x_element = np.array([xi, xj])
        y_element = np.array([yi, yj])

        # Plotting the element
        ax.plot(x_element, y_element, color = 'black', zorder = 0)
        # Plotting the moment diagram
        ax.plot(x_moment, y_moment, color = 'red', linewidth = 2, zorder = 1)
        
        ax.annotate(moment_ini_value + 'N.m',
                    xy=(xi, yi),
                    xytext=(xti, yti), 
                    color = 'red',
                    zorder = 2)
        
        if i == number_elements-1:
            ax.annotate(moment_end_value + 'N.m',
                        xy=(xj, yj),
                        xytext=(xtj, ytj), 
                        color = 'red',
                        zorder = 2)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.show()

def plot_frame_shear_diagram(ENL, element_list, element_internal_forces, element_angles, scale_factor):

    number_elements = element_list.shape[0]

    element_shear_forces = np.zeros([number_elements, 2])
    element_shear_forces[:,0] = element_internal_forces[:,1]
    element_shear_forces[:,1] = element_internal_forces[:,4]

    shear_max = element_shear_forces.max()
    element_scaled_shear_forces = (1/shear_max)*scale_factor*element_shear_forces

    fig, ax = plt.subplots(figsize=(10,10))

    for i in range(0,number_elements):
        # Getting the nodes of the element
        element_nodes = element_list[i,:]
        node_ini = element_nodes[0]
        node_end = element_nodes[1]

        # Node Coordinates
        xi = ENL[node_ini - 1, 0]
        yi = ENL[node_ini - 1, 1]
        xj = ENL[node_end - 1, 0]
        yj = ENL[node_end - 1, 1]

        # Element angle
        beta = element_angles[i]

        # Now we are plotting the scaled shear forces
        shear_ini = element_scaled_shear_forces[i,0]
        shear_end = -element_scaled_shear_forces[i,1] # Negative for plot purposes

        # Initial Shear Coordinates
        xvi = xi - shear_ini*math.sin(beta)
        yvi = yi + shear_ini*math.cos(beta)

        # Final Shear Coordinates
        xvj = xj - shear_end*math.sin(beta)
        yvj = yj + shear_end*math.cos(beta)

        # Text Coordinates
        xti = (xi+xvi)/2
        yti = (yi+yvi)/2
        xtj = (xj+xvj)/2
        ytj = (yj+yvj)/2

        shear_ini_value = str(np.round(element_shear_forces[i,0], decimals=2))
        shear_end_value = str(np.round(element_shear_forces[i,1], decimals=2))

        # Setting the vector of the shear diagram
        x_shear = np.array([xi, xvi, xvj, xj])
        y_shear = np.array([yi, yvi, yvj, yj])

        # Setting the vecotrs of the element
        x_element = np.array([xi, xj])
        y_element = np.array([yi, yj])

        # Plotting the element
        ax.plot(x_element, y_element, color = 'black', zorder = 0)
        # Plotting the moment diagram
        ax.plot(x_shear, y_shear, color = 'green', linewidth = 2, zorder = 1)
        
        ax.annotate(shear_ini_value + 'N',
                    xy=(xi, yi),
                    xytext=(xti, yti), 
                    color = 'green',
                    zorder = 2)
        if i == number_elements-1:
            ax.annotate(shear_end_value + 'N',
                        xy=(xj, yj),
                        xytext=(xtj, ytj), 
                        color = 'green',
                        zorder = 2)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.show()