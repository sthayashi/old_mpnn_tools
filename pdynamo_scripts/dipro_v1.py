import h5py
import numpy as np
from timeit import default_timer as timer

from pCore import Coordinates3
from pMolecule import QCModelMNDO, QCModelDFT, System

from joblib import delayed, Parallel, parallel_backend

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
        return 0, 0, 0

    homo_level = energies[homo] * 27.2113860217
    homo_orbital = np.array(orbitals[:, homo])

    return molecule, homo_level, homo_orbital


###############################################################################
# mol1 is a molecule object from pdynamo
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
#       print "            Dimer ERROR"
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


###############################################################################

def calc_pair(organic_molecule, molecule_homo_orbital, o2pos, O2_HOMO, i):
  # print("entering loop for o2s")
#   for i in range(o2pos.shape[0]):
    each_o2pos = o2pos[i]
    o2_molecule, o2_homo_energy, o2_homo_orbital = compute_molecule(each_o2pos, o2_atom_types)

    t = compute_dimer(organic_molecule, molecule_homo_orbital, o2_molecule, o2_homo_orbital)
    accept = False
    if t == 0.0:
        pass
    elif t == 1e+20:
        pass
    elif abs(t) <= ti_threshold:
        pass
    else:
#       print("Found nonzero t at {}".format(i))
        accept = True
    O2_HOMO[i] = o2_homo_energy
  # print("O2 HOMO SHAPE: {}".format(O2_HOMO.shape))

    return (o2_homo_orbital, o2_homo_energy, t, accept)

def calculate_ti_all_o2(o2pos, organic_homo_energy, organic_molecule, molecule_homo_orbital):
    inds_of_nonzero_integrals = [] 

#   transfer_integrals = np.zeros(o2pos.shape[0])
    organic_HOMO = np.array([organic_homo_energy])
    O2_HOMO = np.empty(o2pos.shape[0])

    print("        Calculating transfer integrals with O2 molecules...")
 
   #some_function(organic_molecule, molecule_homo_orbital, o2pos, O2_HOMO, transfer_integrals, inds_of_nonzero_integrals)
  # print("BEFORE O2 CALCS")
  # print(inds_of_nonzero_integrals)
  # print(transfer_integrals)
    num_o2 = o2pos.shape[0]

    # Use without shared memory should be a lot faster
    result = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(calc_pair)(organic_molecule, molecule_homo_orbital, o2pos, O2_HOMO, i) for i in range(num_o2))
#   with parallel_backend('threading', n_jobs=-1):
#       result = Parallel(n_jobs=-1, prefer="threads")(delayed(calc_pair)(organic_molecule, molecule_homo_orbital, o2pos, O2_HOMO, i) for i in range(num_o2))

   #Parallel(n_jobs=8, require='sharedmem')(delayed(calc_pair)(organic_molecule, molecule_homo_orbital, o2pos, O2_HOMO, transfer_integrals, inds_of_nonzero_integrals, i) for i in range(num_o2))

    inds_of_nonzero_integrals = np.array(inds_of_nonzero_integrals)

    return result

import time

ti_threshold = 1e-25

start = time.time()
with h5py.File(filename, "a") as file:
    keys = list(file.keys())
    for molecule_refcode in keys:
        print("Calculating transfer integrals for {}".format(molecule_refcode))

        # Select an organic molecule and load relevant data
        molecule_group = file.get(molecule_refcode)
        molpos = molecule_group.get("molecule_pos")[()]
        mol_atomic_numbers = molecule_group.get("atomic_numbers")[()]
        atom_types = molecule_group.get("atom_types")[()].astype(str)

        # O2 molecular data
        o2pos = molecule_group.get("o2_pos")[()] # o2_pos includes ALL o2 positions
        o2_atom_types = ['O', 'O']

        # Compute HOMO for the organic molecule once
        print("    Calculating orbitals of organic molecule...")
        organic_molecule, organic_homo_energy, molecule_homo_orbital = compute_molecule(molpos, atom_types)

        result = calculate_ti_all_o2(o2pos, organic_homo_energy, organic_molecule, molecule_homo_orbital)

        o2_homo_orbital = []
        o2_good_inds = []
        assert o2pos.shape[0] == len(result), "Unexpected number of transfer integrals"
        transfer_integrals = np.zeros(o2pos.shape[0], dtype=np.float32)
     
        for i in range(len(result)):
            # Don't save o2 info for now, but it's here if needed
            o2_homo_orbital, o2_homo_energy, ti, accept = result[i]
#           if accept == True:
#               transfer_integrals[i] = ti
#               o2_good_inds.append(i)
            transfer_integrals[i] = result[i][2]
            if result[i][-1] == True:
                o2_good_inds.append(i)

        o2_good_inds = np.array(o2_good_inds)

        # How to overwrite the o2_pos that have bad integrals?
        print("    Saving data...\n")
        molecule_group.require_dataset("transfer_integrals", shape=transfer_integrals.shape, dtype="float32")
        molecule_group.require_dataset("organic_HOMO", shape=molecule_homo_orbital.shape, dtype="float32")
#       molecule_group.require_dataset("O2_HOMO", shape=O2_HOMO.shape, dtype="float32")
        molecule_group.require_dataset("o2_good_inds", shape=o2_good_inds.shape, dtype="int")
        
        molecule_group["transfer_integrals"][:] = transfer_integrals
        molecule_group["organic_HOMO"][:] = molecule_homo_orbital
#       molecule_group["O2_HOMO"][:] = O2_HOMO
        molecule_group["o2_good_inds"][:] = o2_good_inds
end = time.time()
print("Time taken: {}".format(end-start))
