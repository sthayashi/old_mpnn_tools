import h5py
import numpy as np
from timeit import default_timer as timer

from multiprocessing import Pool, cpu_count

from pCore import Coordinates3
from pMolecule import QCModelMNDO, QCModelDFT, System

from joblib import Parallel, delayed

#from pMoleculeScripts import MergeByAtom

mndo = QCModelMNDO("rm1", isSpinRestricted=True, keepOrbitalData=True)
dft = QCModelDFT(functional="lda", orbitalBasis="sto3g",
                 isSpinRestricted=True, keepOrbitalData=True)
qcmodel = mndo

#filename = "Data/full_DNTT.h5"
#num_procs = 32
filename = "parsed_molecules.h5"
num_procs = 2


###############################################################################
def numpy2coordinates(positions):
    coords = Coordinates3.WithExtent(len(positions))
    for i in range(len(positions)):
        coords[i, 0] = positions[i, 0]
        coords[i, 1] = positions[i, 1]
        coords[i, 2] = positions[i, 2]
    return coords

def coordinates2numpy(mol):
    return np.array([mol.coordinates3[i] for i in range(np.shape(mol.coordinates3)[0])])

###############################################################################
def compute_molecule(molpos, atom_types):
    molecule = System.FromAtoms(atom_types)
    molecule.coordinates3 = numpy2coordinates(molpos)
    molecule.DefineQCModel(qcmodel)

    try:
        molecule.Energy(log=None, doGradients=True)
        (energies, homo, lumo) = molecule.energyModel.qcModel.OrbitalEnergies(molecule.configuration)
        orbitals = molecule.energyModel.qcModel.Orbitals(molecule.configuration)
    except:
        print("Molecule ERROR")
        return 0, 0, 0

    homo_level = energies[homo] * 27.2113860217
    homo_orbital = np.array(orbitals[:, homo])

    return molecule, homo_level, homo_orbital


###############################################################################
def compute_dimer(mol1, organic_molecule_homo, mol2, o2_molecule_homo):
    mol1_atomic_numbers = [mol1._atoms[i].atomicNumber for i in range(len(mol1._atoms))]
    mol2_atomic_numbers = [mol2._atoms[i].atomicNumber for i in range(len(mol2._atoms))]

    dimer_atom_types = np.concatenate((mol1_atomic_numbers, mol2_atomic_numbers))
    dimer_molecule = System.FromAtoms(dimer_atom_types)
    dimer_coords = np.concatenate((coordinates2numpy(mol1), coordinates2numpy(mol2)))
    dimer_molecule.coordinates3 = numpy2coordinates(dimer_coords)

    dimer_molecule.DefineQCModel(qcmodel)

    try:
        dimer_molecule.Energy(log=None, doGradients=True)
        (energies12, homo12, lumo12) = dimer_molecule.energyModel.qcModel.OrbitalEnergies(dimer_molecule.configuration)
        dimer_orbitals = dimer_molecule.energyModel.qcModel.Orbitals(dimer_molecule.configuration)
    except:
        print "            Dimer ERROR"
        return 1e20

    orb1_size = organic_molecule_homo.shape[0]
    orb12_size = dimer_orbitals.shape[0]
    orbital1 = np.zeros(orb12_size)
    orbital2 = np.zeros(orb12_size)
    orbital12 = np.zeros((orb12_size, orb12_size))

    orbital1[:orb1_size] = organic_molecule_homo
    orbital2[orb1_size:] = o2_molecule_homo

    for r in range(dimer_orbitals.rows):
        orbital12[r] = dimer_orbitals[r, :]

    gamma1 = np.matmul(orbital1, orbital12)
    gamma2 = np.matmul(orbital2, orbital12)
    e1 = gamma1.dot(energies12 * gamma1)
    e2 = gamma2.dot(energies12 * gamma2)
    J12 = gamma1.dot(energies12 * gamma2) #check
    S12 = gamma1.dot(gamma2)

    J12_eff = (J12 - S12 * (e1 + e2) / 2) / (1 - S12**2) * 27.2113860217

    return J12_eff

#def compute_ti(molpos, mol_atomic_numbers, atom_types, o2pos):
def compute_ti(group_data):
    molpos, mol_atomic_numbers, atom_types, o2pos = group_data
    o2_atom_types = ['O', 'O']

    # Compute HOMO for the organic molecule once
    print("    Calculating orbitals of organic molecule...")
    organic_molecule, organic_homo_energy, molecule_homo_orbital = compute_molecule(molpos, atom_types)

    inds_of_nonzero_integrals = [] 

    # preallocate returned arrays
    transfer_integrals = np.zeros(o2pos.shape[0], dtype=np.float32)
    organic_HOMO = np.array([organic_homo_energy])
    O2_HOMO = np.empty(o2pos.shape[0])

    print("        Calculating transfer integrals with O2 molecules...")

    for i in range(o2pos.shape[0]):
        each_o2pos = o2pos[i, :, :]
        o2_molecule, o2_homo_energy, o2_homo_orbital = compute_molecule(each_o2pos, o2_atom_types)

        t = compute_dimer(organic_molecule, molecule_homo_orbital, o2_molecule, o2_homo_orbital)
        if t == 0.0:
            continue
        elif t == 1e+20:
            continue
        elif abs(t) <= ti_threshold:
            continue
        else:
            print("Found nonzero t")
            inds_of_nonzero_integrals.append(i)
            transfer_integrals[i] = t
        O2_HOMO[i] = o2_homo_energy

    inds_of_nonzero_integrals = np.array(inds_of_nonzero_integrals)

    return transfer_integrals, inds_of_nonzero_integrals

###############################################################################
# Group names are csd refcodes.
file = h5py.File(filename, "a")
keys = list(file.keys())

ti_threshold = 1e-25

import time
start = time.time()

all_molecule_groups = [file.get(refcode) for refcode in keys]
all_molecule_group_data = []
for group in all_molecule_groups:
    organic_pos = group.get('molecule_pos')[:]
    mol_atomic_numbers = group.get("atomic_numbers")[()]
    atomic_types = group.get("atom_types")[:]
    o2_pos = group.get('o2_pos')[:]

    all_molecule_group_data.append([organic_pos, mol_atomic_numbers, atomic_types, o2_pos])

ti_threshold = 1e-25

pool = Pool(processes=cpu_count())

result = pool.map(compute_ti, all_molecule_group_data)

for i in range(len(result)):
    molecule_group = all_molecule_groups[i]

    transfer_integrals = result[i][0]
    inds_of_nonzero_integrals = result[i][1]

    # How to overwrite the o2_pos that have bad integrals?
    print("    Saving data...\n")
    molecule_group.require_dataset("transfer_integrals", shape=transfer_integrals.shape, dtype="float32")
    molecule_group.require_dataset("o2_good_inds", shape=inds_of_nonzero_integrals.shape, dtype="int")
#   molecule_group.require_dataset("organic_HOMO", shape=organic_HOMO.shape, dtype="float32")
#   molecule_group.require_dataset("O2_HOMO", shape=O2_HOMO.shape, dtype="float32")
    
    molecule_group["transfer_integrals"][:] = transfer_integrals
    molecule_group["o2_good_inds"][:] = inds_of_nonzero_integrals
#   molecule_group["organic_HOMO"][:] = organic_HOMO
#   molecule_group["O2_HOMO"][:] = O2_HOMO

file.close()
pool.close()

end = time.time()
print("Time taken: {}".format(end-start))
