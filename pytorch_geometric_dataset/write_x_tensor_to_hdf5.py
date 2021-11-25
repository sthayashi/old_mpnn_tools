# note for self: I left off in runs/0*
# if the runs worked, it should now how organic_HOMO and O2_HOMO.
# no more calculations should be needed to start training.

import h5py
import numpy as np
"""
The purpose of this script is to take the output from parse_df.py -> generate_o2_pos.py
and generate the needed mol_new.h5 file, which is used by 2_initialize.py.

2_initialize.py crates the dataset used in the GNN.

TODO: Add and confirm O2_HOMO is supposed to actually be the energy.
TODO: Wrap in a for loop when I understand how the pytorch geometric dataset stacks
"""

with h5py.File("parsed_molecules.h5", "a") as f:
    keys = list(f.keys())
    for k in keys:
        molecule = f.get(k)
        print(molecule)

        # Data that doesn't need reshaping or manipulation
        coulomb_interaction = molecule.get("coulomb_interaction")[:]

        # Data from parse_df.py -> generate_o2_pos.py, which needs to be combined to coform to what the workflow expects
        atom_types = molecule.get("atom_types")[:]
        atomic_numbers = molecule.get("atomic_numbers")[:]

        # Put into 2d array, with first column as molecule number (1 for organic, 2 for O2)
        # and second column as the atomic number

        # First the organic molecule
        molecule_numbers = np.ones(len(atomic_numbers))
        
        # Then the O2 molecule
        molecule_numbers = np.append(molecule_numbers, [2., 2.])
        atomic_numbers = np.append(atomic_numbers, [8., 8.])

        x = np.vstack((molecule_numbers, atomic_numbers)).transpose()

        molecule.require_dataset("x", shape=x.shape, dtype="float32")
        molecule["x"][:] = x
