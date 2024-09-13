import numpy as np
# ----------------------------------------------------------------------------
# INPUT DATA
# ----------------------------------------------------------------------------

def get_frame_input_data(problem):
    # ----------------------------------------------------------------------------
    # NODE LIST
    # ----------------------------------------------------------------------------
    # [x,y]
    node_list_txt = './Problems/'+problem +'_node_list.txt'
    node_list = np.loadtxt(node_list_txt, delimiter=',')

    # ----------------------------------------------------------------------------
    # NODE RESTRAINTS
    # ----------------------------------------------------------------------------
    # -1: Fixed
    # +1: Free
    # [x,y,rz]
    # ----------------------------------------------------------------------------
    node_restraints_txt = './Problems/'+problem +'_node_restraints.txt'
    node_restraints = np.loadtxt(node_restraints_txt, delimiter=',')

    # ----------------------------------------------------------------------------
    # ELEMENT LIST
    # ----------------------------------------------------------------------------
    element_list_txt = './Problems/'+problem +'_element_list.txt'
    element_list = np.loadtxt(element_list_txt, dtype=int, delimiter=',')

    # ----------------------------------------------------------------------------
    # CROSS-SECTION PROPERTIES
    # ----------------------------------------------------------------------------
    # Element Proprerty [Ai, Ii, Ei]
    element_properties_txt = './Problems/'+problem +'_section_properties.txt'
    element_properties = np.loadtxt(element_properties_txt, delimiter=',')
    
    # ----------------------------------------------------------------------------
    # NODE FORCES Fu
    # ----------------------------------------------------------------------------
    # [Fx, Fy, Mz]
    node_forces_txt = './Problems/'+problem +'_node_forces.txt'
    Fu = np.loadtxt(node_forces_txt, delimiter=',')

    # ----------------------------------------------------------------------------
    # ELEMENT DISTRIBUTED LOADS
    # ----------------------------------------------------------------------------
    element_distributed_loads_txt = './Problems/'+problem +'_element_distributed_loads.txt'
    element_distributed_loads = np.loadtxt(element_distributed_loads_txt, delimiter=',')

    # ----------------------------------------------------------------------------
    # DISPLACEMENTS [U_u]
    # ----------------------------------------------------------------------------
    # Initially zeros
    node_displacements_txt = './Problems/'+problem +'_node_displacements.txt'
    U_u = np.loadtxt(node_displacements_txt, delimiter=',')

    return (node_list, node_restraints, element_list, element_properties, Fu, element_distributed_loads, U_u)