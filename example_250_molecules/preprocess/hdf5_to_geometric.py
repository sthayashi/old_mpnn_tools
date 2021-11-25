# note for self: I left off in runs/0*
# if the runs worked, it should now how organic_HOMO and O2_HOMO.
# no more calculations should be needed to start training.

import h5py
import numpy as np
import mendeleev
from itertools import permutations, product

"""
This script generates the node features x for an organic+O2 molecule pair.
Note this isn't required for every iterative pair.

TODO: Add and confirm O2_HOMO is supposed to actually be the energy.
TODO: Wrap in a for loop when I understand how the pytorch geometric dataset stacks
"""

def make_node_features(atomic_numbers):
    """
    Create node features

    Input
    atomic_numbers: Periodic table

    Return
    x: The node features used by pytorch geometric
    """
    # Atoms belonging to organic molecules
    molecule_memberships = np.ones(len(atomic_numbers))

    # Atoms belonging to O2
    molecule_memberships = np.append(molecule_memberships, [2., 2.])

    electronegativities = np.empty(atomic_numbers.shape, dtype=float)
    for i in range(electronegativities.shape[0]):
        electronegativities[i] = mendeleev.element(int(atomic_numbers[i])).electronegativity()
    electronegativities = np.append(electronegativities, [3.44, 3.44])

    return np.vstack((molecule_memberships, electronegativities)).transpose()

def flatten_coulomb_interactions(coulomb_interactions):
    """
    Put coulomb interactions into pytorch geomtric's edge_attributes format,
    assuming the graph is bidirectional.

    Input
    coulomb_interactions: A (NxN) array. 
                          (i,j) is the coulomb interaction between atom i and atom j.
                          E.g. H2O2 is a (4x2) array.

    Return
    parsed_ci: (NX2) array. The "parsed", or pytorch-geometric-formatted, coulomb interactions.
               E.g. For H2O2, return a (13,) array.
    """
    pass

def calculate_Cij(i, Z_i, R_i, j, Z_j, R_j):
    if i == j:
        print("THIS SHOULDN'T HAPPEN")
        return 0.5 * Z_i**2.4
    else:
        return (Z_i*Z_j)/np.linalg.norm(R_i-R_j)

def make_edge_indices(organic_coords, o2_coords, include_intermolecular=True):
    """
    Create the edge indices and features of a directional graph
    between the organic and O2 molecules.
    """
    num_organic_atoms = organic_coords.shape[0]
    organic_indices = np.array([i for i in range(num_organic_atoms)])
    o2_indices = np.array([num_organic_atoms, num_organic_atoms+1])

    edge_indices_1 = []
    edge_indices_2 = []

    if include_intermolecular == True:
        indices = np.concatenate((organic_indices, o2_indices))
        for i,j in permutations(indices, 2):
            edge_indices_1.append(i)
            edge_indices_2.append(j)

        edge_indices_1 = np.array(edge_indices_1)
        edge_indices_2 = np.array(edge_indices_2)
    else:
        for i,j in product(organic_indices, o2_indices):
            edge_indices_1.append(i)
            edge_indices_2.append(j)

    return (edge_indices_1, edge_indices_2)

def make_edge_features(edge_indices_1, edge_indices_2, atomic_numbers, dimer_coords):
    """
    Inputs:
    edge_indices_*: 1D arrays which refer to edges in the graph. E.g.
                    [1, 2], [3, 4] means edges are between (1,3) and (2, 4).

    atomic_numbers: 1D array. 
                    The atomic numbers of organic+O2 molecules concatenated.
                    Ordered with dimer_coords. 

    dimer_coords:   (number_of_atomsx3) array.
                    All coordinates of all molecules. Order of atoms assumed to match edge_indices.
                    dimer_coords should be the concatenation of organic+O2 molecules.
                    For example, with H2O2, we could have edge indices [1, 2] for H2, 
                    and edge indices [3, 4] for O2. We calculate the coulomb interaction
                    for atoms [1, 3] and [2, 4].

    Returns:
    coulomb_interactions: Coulomb interactions as a 1D array, in order of edge_indices_*
    """
    coulomb_interactions = []
    # NOTE: I'm assuming zip keeps ordering. This is important, because we are directly indexing the molecules.
    for i,j in zip(edge_indices_1, edge_indices_2):
        ci = calculate_Cij(i, atomic_numbers[i], dimer_coords[i], j, atomic_numbers[j], dimer_coords[j])
        coulomb_interactions.append(ci)
    return np.array(coulomb_interactions)

with h5py.File("parsed_molecules.h5", "a") as f:
    keys = list(f.keys())
    for k in keys:
        molecule = f.get(k)

        o2_good_inds = molecule.get("o2_good_inds")[:]
        
        mol_pos = molecule.get("molecule_pos")[:]
        o2_pos = molecule.get("o2_pos")[:][o2_good_inds]

        # Data from parse_df.py -> generate_o2_pos.py, which needs to be combined to coform to what the workflow expects
        atom_types = molecule.get("atom_types")[:]
        atomic_numbers = molecule.get("atomic_numbers")[:]

        # Node features x is a 2d array, with first column representing molecular membership (1 for organic, 2 for O2)
        # and second column as the atomic number
        x = make_node_features(atomic_numbers)

        # edges
        edge_indices_i, edge_indices_j = make_edge_indices(mol_pos, o2_pos, True)

        # edge features
        num_edges = np.array(edge_indices_i).shape[0]
        num_edge_features = 1
        coulomb_interactions = np.zeros((o2_pos.shape[0], num_edges), np.float32)
        dimer_atomic_numbers = np.concatenate((atomic_numbers, [8, 8]))
        for i in range(coulomb_interactions.shape[0]):
            dimer_coords = np.concatenate((mol_pos, o2_pos[i, :, :]))
            ci = make_edge_features(edge_indices_i, edge_indices_j, dimer_atomic_numbers, dimer_coords)
            coulomb_interactions[i, :] = ci

        assert len(edge_indices_i) == coulomb_interactions.shape[1], "Number of edge features not equal to number of edges!"

        edge_indices_i = np.array(edge_indices_i, dtype=int)
        edge_indices_j = np.array(edge_indices_j, dtype=int)


        molecule.require_dataset("x", shape=x.shape, dtype="float32")

        molecule.require_dataset("edge_indices_i", shape=edge_indices_i.shape, dtype="int")
        molecule.require_dataset("edge_indices_j", shape=edge_indices_j.shape, dtype="int")

        molecule.require_dataset("coulomb_interactions", shape=coulomb_interactions.shape, dtype="float32")

        molecule["x"][:] = x
        molecule["edge_indices_i"][:] = edge_indices_i
        molecule["edge_indices_j"][:] = edge_indices_j
        molecule["coulomb_interactions"][:] = coulomb_interactions 
