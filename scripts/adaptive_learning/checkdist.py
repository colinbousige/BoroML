from ase import Atoms
import numpy as np


#------------------    ALLOW TO CHECK DISTANCES BETWEEN ATOMS ON STRUCTURES  -------------------------

# minimum distances
dHH = 0.65
dHO = 0.75
dHAlSi = 0.9
dOAlSi = 1.1
dOO = 1.0
dAlAlSiSi = 1.3

def check_distances(struct):
    symbols = struct.get_chemical_symbols()
    H_indices = [index for index, element in enumerate(symbols) if element == 'H']
    O_indices = [index for index, element in enumerate(symbols) if element == 'O']
    Al_indices = [index for index, element in enumerate(symbols) if element == 'Al']
    Si_indices = [index for index, element in enumerate(symbols) if element == 'Si']
    Al_Si_indices = Al_indices + Si_indices
    for h in H_indices:
        dist = struct.get_distances(h, H_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dHH)):
            return False
        dist = struct.get_distances(h, O_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dHO)):
            return False
        dist = struct.get_distances(h, Al_Si_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dHAlSi)):
            return False
    for o in O_indices:
        dist = struct.get_distances(o, Al_Si_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dOAlSi)):
            return False
        dist = struct.get_distances(o, O_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dOO)):
            return False
    for alsi in Al_Si_indices:
        dist = struct.get_distances(alsi, Al_Si_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dAlAlSiSi)):
            return False
    return True


def check_distances_verb(struct):
    symbols = struct.get_chemical_symbols()
    H_indices = [index for index, element in enumerate(symbols) if element == 'H']
    O_indices = [index for index, element in enumerate(symbols) if element == 'O']
    Al_indices = [index for index, element in enumerate(symbols) if element == 'Al']
    Si_indices = [index for index, element in enumerate(symbols) if element == 'Si']
    Al_Si_indices = Al_indices + Si_indices
    for h in H_indices:
        dist = struct.get_distances(h, H_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dHH)):
            print("dHH: ",np.sort(dist)[:5])
            return False
        dist = struct.get_distances(h, O_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dHO)):
            print("dOH: ",np.sort(dist)[:5])
            return False
        dist = struct.get_distances(h, Al_Si_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dHAlSi)):
            print("dHAlSi: ",np.sort(dist)[:5])
            return False
    for o in O_indices:
        dist = struct.get_distances(o, Al_Si_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dOAlSi)):
            print("dOAlSi: ",np.sort(dist)[:5])
            return False
        dist = struct.get_distances(o, O_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dOO)):
            print("dOO: ",np.sort(dist)[:5])
            return False
    for alsi in Al_Si_indices:
        dist = struct.get_distances(alsi, Al_Si_indices, mic=True)
        if np.isin(True, np.logical_and(dist>0.00001, dist<dAlAlSiSi)):
            print("dAlSiAlSi: ",np.sort(dist)[:5])
            return False
    return True
