import numpy as np
import swiftsimio as sw
import warnings

def get_data(file, target_ids = np.empty(0), N = 0, a = True, box = False):   
    """ 
        Function to read data 

        file: path + filename
        target_ids: list of particle id's you're interested in. If left blank, we will look at particles with ids 0, 1, 2, ..., N-1
        N: number of particles you'd like to have from the sample. If N = 0, all the particles will be used.
        a: True multiplies coordinates by a
        box: True returns boxsize
    """ 
    # reading in the data
    data = sw.load(file)

    # getting the info about particle coordinates, id and boxsize
    ids = data.dark_matter.particle_ids
    coordinates = data.dark_matter.coordinates
    boxsize = data.metadata.boxsize
    at = data.metadata.a if a else 1
    
    sort_order = np.argsort(ids)
    sorted_ids = ids[sort_order]
    sorted_coords = coordinates[sort_order] * at

    indices = np.empty(0)
    if len(target_ids) != 0:
        target_ids = np.sort(target_ids)

        # getting the indices of the target ids
        with warnings.catch_warnings(action="ignore"):
            indices = [np.where(sorted_ids == i)[0][0] for i in target_ids]
        p = sorted_coords[indices]

    elif N != 0:
        p = sorted_coords[:N] 
    
    else: 
        p = sorted_coords 

    p = np.array(p)

    if box:
        return p, np.array(boxsize)
    
    return p