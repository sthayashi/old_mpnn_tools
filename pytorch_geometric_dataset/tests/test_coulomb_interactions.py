import numpy as np
import mendeleev

import h5py

with h5py.File("parsed_molecules.h5", "r") as f:
    keys = list(f.keys())

    for refcode in keys:
        molecule = f.get(refcode)

        organic_pos = molecule.get("molecule_pos")[:]
        organic_atom_types = molecule.get("atom_types")[:].astype(str)

        o2_good_inds = molecule.get("o2_good_inds")[:]
        o2_pos = molecule.get("o2_pos")[:][o2_good_inds, :, :]
        o2_atom_types = ['O', 'O']

        edge_indices_i = molecule.get("edge_indices_i")[:]
        edge_indices_j = molecule.get("edge_indices_j")[:]
        coulomb_interactions = molecule.get("coulomb_interactions")[:]

        # Test edges : This is just over the first organic molecule
        for o2_ind in range(o2_pos.shape[0]):
            one_o2_coord = o2_pos[o2_ind, :, :]
            ci = coulomb_interactions[o2_ind, :]
    
            Z_j = 8.
            i_max = max(edge_indices_i)
            ci_index = 0
            for i,j in list(zip(edge_indices_i, edge_indices_j)):
                atom_i = organic_atom_types[i]
                Z_i = float(mendeleev.element(atom_i).atomic_number)
                
                R_i = organic_pos[i]
                R_j = one_o2_coord[j-i_max-1]
                R_ij = np.linalg.norm(R_i - R_j)
    
                expected_ci = Z_i * 8. / R_ij
    
                is_correct = np.isclose(expected_ci, ci[ci_index])
                if is_correct == False:
                    print("Failed!")
                ci_index += 1
