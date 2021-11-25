import numpy as np
import pandas as pd
import mendeleev
import h5py

def parse_molecule_df_xyz(molecule_df):
    xyz_str = molecule_df.xyz_pbe_relaxed.strip().split()

    number_of_atoms = int(xyz_str.pop(0))
    assert number_of_atoms*3 + number_of_atoms == len(xyz_str), "Unexpected number of array elements"

    atom_types = np.empty(number_of_atoms, dtype="S2")
    atom_coords = np.empty((number_of_atoms, 3), dtype=np.float)

    # Break up into chunks of atom types and their corresponding coordinates
    atom_chunks = [xyz_str[i * 4:(i + 1) * 4] for i in range((len(xyz_str) + 4 - 1) // 4 )]
    for i in range(len(atom_chunks)):
        atom_types[i] = atom_chunks[i][0]
        atom_coords[i] = atom_chunks[i][1:]

    return (atom_types, atom_coords)

def get_atomic_numbers(atom_types):
    atomic_numbers = np.empty(len(atom_types), dtype=np.int)
    for i in range(len(atomic_numbers)):
        atom_type = atom_types[i].decode()
        atomic_numbers[i] = mendeleev.element(atom_type).atomic_number
    return atomic_numbers

FILEPATH = "/mnt/projects/sne_students/M.Thesis_Scott/data/m1507656/df_62k.json"

df = pd.read_json(FILEPATH, orient='split')

df = df.sort_values("number_of_atoms")
rand_ints = np.unique(np.random.randint(int(len(df)/4), int(len(df)/2), size=20))
df_chunk = df.iloc[rand_ints]

with h5py.File("parsed_molecules.h5", 'w') as file_out:
    for i in range(len(df_chunk)):

        molecule_df = df_chunk.iloc[i]
        atom_types, molecule_pos = parse_molecule_df_xyz(molecule_df)
        molecule_pos -= np.mean(molecule_pos, axis=0)
        atomic_numbers = get_atomic_numbers(atom_types)
        # Si doesn't work with pdynamo
        if 14 in atomic_numbers:
            print("Skipping refcode {}".format(molecule_df.refcode_csd))
            continue

        smiles = np.array(molecule_df.canonical_smiles.encode("utf8"))
        refcode_csd = molecule_df.refcode_csd

        group = file_out.require_group(refcode_csd)
        group.require_dataset("molecule_pos", shape=molecule_pos.shape, dtype=molecule_pos.dtype)
        group.require_dataset("atom_types", shape=atom_types.shape, dtype=atom_types.dtype)
        group.require_dataset("atomic_numbers", shape=atomic_numbers.shape, dtype=atomic_numbers.dtype)
        group.require_dataset("smiles", shape=np.shape(smiles), dtype=h5py.string_dtype())

        group["molecule_pos"][:] = molecule_pos
        group["atom_types"][:] = atom_types
        group["atomic_numbers"][:] = atomic_numbers
        group["smiles"][()] = smiles
