import numpy as np

import h5py

with h5py.File("parsed_molecules.h5", 'r') as f:
    keys = list(f.keys())
    for refcode in keys:
        group = f.get(refcode)
    
        o2_good_inds = group.get("o2_good_inds")[:]
    
        # coordinates
        o2_pos = group.get("o2_pos")[o2_good_inds, :, :]
        organic_pos = group.get("molecule_pos")[:]
    
        # atomic info
        atomic_numbers = group.get("atomic_numbers")[:]
        dimer_atomic_numbers = np.concatenate((atomic_numbers, [8, 8]))
    
        # nodes
        x = group.get("x")[:]

        # edges
        edge_indices_i = group.get("edge_indices_i")[:]
        edge_indices_j = group.get("edge_indices_j")[:]
        edge_attr = group.get("edge_attr")[:]

        i = 10
        j = edge_indices_j[i]
        Z_i = atomic_numbers[i]
        Z_j = 8


        # testing
        some_o2_ind = 132
        some_o2_pos = o2_pos[some_o2_ind, :, :]

        # check coulomb matrix element
        Rij = np.linalg.norm(organic_pos[i] - some_o2_pos[j - len(organic_pos)])
        Cij = (Z_i * Z_j) / Rij

        np.testing.assert_approx_equal(Cij, edge_attr[some_o2_ind, :, :][i][0], 6)

        # check node features
       #print(np.shape(x))
