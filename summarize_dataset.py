import h5py
import numpy as np
import pandas as pd

filename1 = "parsed_molecules.h5"
with h5py.File(filename1, "r") as file:
    keys = list(file.keys())
    number_of_organic_molecules = len(keys)

    refcode = []
    number_of_atoms = []
    number_of_valid_o2 = []

    for molecule_refcode in keys:
        refcode.append(molecule_refcode)
        molecule_group = file.get(molecule_refcode)

        molpos = molecule_group.get("molecule_pos")[()]
        o2_pos = molecule_group.get("o2_pos")[()]
        o2_good_inds = molecule_group["o2_good_inds"][:]

        mol_atomic_numbers = molecule_group.get("atomic_numbers")[()]
        atom_types = molecule_group.get("atom_types")[()].astype(str)

        number_of_atoms.append(len(atom_types))
        number_of_valid_o2.append(len(o2_good_inds))

for i in range(number_of_organic_molecules):
    print("CSD Refcode: {}".format(refcode[i]))
    print("Number of Atoms: {}".format(number_of_atoms[i]))
    print("Number of training samples: {}".format(number_of_valid_o2[i]))
    print()


#df = pd.DataFrame(list(zip(refcode, number_of_atoms, number_of_valid_o2)), columns=['refcode', 'num_atoms', 'num_train_smpls'])
#df.to_csv("dataset_summary.csv")
