import h5py
import numpy as np
from scipy.spatial import distance_matrix
from SALib.sample import saltelli, latin
from pyquaternion import Quaternion


################################################################################
# helper functions

# calculate quaternion from Euler angles  z - y' - x''  convention
def euler2quat(alpha, beta, gamma, ang="rad"):
    if ang == "deg":
        alpha *= np.pi / 180
        beta *= np.pi / 180
        gamma *= np.pi / 180
    elif ang == "rad":
        pass
    else:
        raise TypeError("ang input is not in [\"rad\", \"deg\"]")

    rot_alpha = np.eye(3)
    rot_alpha[0, 0] = np.cos(alpha)
    rot_alpha[1, 1] = rot_alpha[0, 0]
    rot_alpha[1, 0] = np.sin(alpha)
    rot_alpha[0, 1] = -rot_alpha[1, 0]
    rot_beta = np.eye(3)
    rot_beta[0, 0] = np.cos(beta)
    rot_beta[2, 2] = rot_beta[0, 0]
    rot_beta[2, 0] = np.sin(beta)
    rot_beta[0, 2] = -rot_beta[2, 0]
    rot_gamma = np.eye(3)
    rot_gamma[1, 1] = np.cos(gamma)
    rot_gamma[2, 2] = rot_gamma[1, 1]
    rot_gamma[1, 2] = np.sin(gamma)
    rot_gamma[2, 1] = -rot_gamma[1, 2]

    rotmat = np.matmul(rot_alpha, np.matmul(rot_beta, rot_gamma))
    return Quaternion(matrix=rotmat)


# translate polar coordinates to cartesian coordinates
def polar2cartesian(distance, ang="rad"):
    if not isinstance(distance, np.ndarray) or not distance.shape == (3, ):
        raise TypeError("distance input is not a 3d vector")
    if ang not in ["rad", "deg"]:
        raise TypeError("ang input is not in [\"rad\", \"deg\"]")

    if ang == "deg":
        distance[1:] *= np.pi / 180

    x = distance[0] * np.cos(distance[1]) * np.cos(distance[2])
    y = distance[0] * np.sin(distance[1]) * np.cos(distance[2])
    z = distance[0] * np.sin(distance[2])

    return np.array([x, y, z])

# Functions for parsing and checking proposal o2


"""
    Parses out the atoms and their corresponding xyz coordinates.

    Input: A single group from an h5 file.
           File must have datasets "molecule_pos" and "atom_types".
           These datasets are assumed to be numpy arrays (see df_to_xyz.py)

    Output: atom_types   :: numpy array of eltype str
            molecule_pos :: numpy array of corresponding xyz coordinates
"""
def parse_h5_group(group, return_o2=False):
    molecule_pos = group.get("molecule_pos")[()]
    atom_types = group.get("atom_types")[()].astype(str)
    atomic_numbers = group.get("atomic_numbers")[()].astype(np.int)
    if return_o2 == False:
        return (atom_types, atomic_numbers, molecule_pos)
    else:
        return (atom_types, atomic_numbers, molecule_pos, o2_pos)


"""
    Checks for any problems before accepting an o2 position.

    Input: molecule_pos :: numpy array of a molecule's xyz coordinates.
           o2_pos       :: numpy array of proposed o2 xyz coordinates.
           r_min        :: minimum distance to accept
           r_max        :: maximum distance to accept

    Output: Boolean value for accepting or rejecting proposed o2 position

    TODO: atom type dependency for allowable distances?
    TODO: I think r_min and r_max shouldn't be needed. 
          It should be handled when generating samples.
"""
def accept_o2(molecule_pos, o2_pos, r_min=0.0, r_max=np.Inf, rtol_scaling=1):
    rtol = 1.0
    accept_o2 = True
 
    # distance Check
    dist_mat = distance_matrix(molecule_pos, o2_pos)

    if True in (dist_mat < r_min):
        accept_o2 = False
    elif True in (dist_mat > r_max):
        accept_o2 = False

    return accept_o2

# Calaculate intramolecular coulomb matrix
def calculate_intramolecular_cm(atomic_numbers, molecule_pos):
    N = len(atomic_numbers)
    M_intramolecular = np.empty((N,N), dtype=molecule_pos.dtype)

    for i in range(N):
        Z_i = atomic_numbers[i]
        R_i = molecule_pos[i]
        for j in range(N):
            Z_j = atomic_numbers[j]
            R_j = molecule_pos[j]
            if i==j:
                M_intramolecular[i,j] = 0.5 * Z_i**2.4
            else:
                M_intramolecular[i,j] = (Z_i*Z_j)/np.linalg.norm(R_i-R_j)
    return M_intramolecular

# Calculate coulomb matrix between o2 and another molecule
def calculate_intermolecular_cm(atomic_numbers, molecule_pos, o2_pos):
    M_intermolecular = np.empty((len(atomic_numbers), 2), dtype=np.float)

    Z_j = 8
    for i in range(len(atomic_numbers)):
        Z_i = int(atomic_numbers[i])
        R_i = molecule_pos[i]

        for j in range(len(o2_pos)):
            R_j = o2_pos[j]
            M_intermolecular[i,j] = (Z_i*Z_j)/np.linalg.norm(R_i-R_j)
    
    return M_intermolecular

# Calculate coulomb matrix of o2 and another molecule
def calculate_coulomb_matrix(atomic_numbers, molecule_pos, R_o2):
    cm_inter = calculate_intermolecular_cm(atomic_numbers, molecule_pos, R_o2)
    cm_intra = calculate_intramolecular_cm(atomic_numbers, molecule_pos)
    return np.hstack([cm_intra, cm_inter])
        

################################################################################
# script


# open h5 file and read positions and atom types of molecule
h5_fname = "parsed_molecules.h5"
with h5py.File(h5_fname, "a") as file_in:
    group_names = list(file_in.keys())

    # define O2 molecule. Bond length ~ 1.21 Angstrom
    o2_base = np.array([[-0.605, 0, 0], [0.605, 0, 0]])

    for molecule in group_names:
        print("Generating o2 positions for {}".format(molecule))
        group = file_in.get(molecule)
        atom_types, atomic_numbers, molecule_pos = parse_h5_group(group)

        molecule_mean = np.mean(molecule_pos)
        molecule_std = np.std(molecule_pos)

        radii = np.linalg.norm(molecule_pos, axis=1)
        r_mean = np.mean(radii)
        r_std = np.std(radii)
        r_min = np.min(radii)
        r_max = np.max(radii)

        r_bounds = [r_max + 0.5, r_max + 5.0]

        problem = {
            "num_vars": 5,
            "names": ["r", "theta", "phi", "alpha", "beta"],
            "bounds": [r_bounds, [-np.pi, np.pi], [-np.pi / 2, np.pi / 2],
                       [-np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2]]
        }

        samples = 5000
        
        N = samples // (problem["num_vars"] + 2)
        X = saltelli.sample(problem, N, False)

        print("sample success")

        o2_pos = np.zeros((X.shape[0], 2, 3))

        for i, params in enumerate(X):
            shift = polar2cartesian(params[:3])  # saltelli returns polar coordinates
            quat = euler2quat(params[3], params[4], 0) # alpha and beta to quaternion
        
            # Transform starting o2 molecule. Shift is translation (+ rotation?) and quat handles yaw and pitch.
            o2_pos[i] = np.array([quat.rotate(v) + shift for v in o2_base]) # List comprehension of both oxygens

        accept = [accept_o2(molecule_pos, o2_pos[i]) for i in range(o2_pos.shape[0])]
        o2_pos = o2_pos[accept, :, :]

        print("="*100)
                             
        coulomb_interaction = np.empty((o2_pos.shape[0], len(atomic_numbers), 2), dtype=np.float)
        for i in range(coulomb_interaction.shape[0]):
            coulomb_interaction[i, :, :] = calculate_intermolecular_cm(atomic_numbers, molecule_pos, o2_pos[i])
       #[coulomb_interaction[i, :, :] = calculate_intermolecular_cm(atomic_numbers, molecule_pos, o2_pos[i]) for i in range(len(o2_pos))]
       #coulomb_interaction = [calculate_intermolecular_cm(atomic_numbers, molecule_pos, o2_pos[i]) for i in range(len(o2_pos))]
        print(coulomb_interaction.shape)
      # cm_inter = calculate_intermolecular_cm(atomic_numbers, molecule_pos, o2_pos[0])
      # cm_intra = calculate_intramolecular_cm(atomic_numbers, molecule_pos)

        group.require_dataset("o2_pos", shape=o2_pos.shape, dtype="float32")
        group.require_dataset("coulomb_interaction", shape=coulomb_interaction.shape, dtype=coulomb_interaction.dtype)

        group["o2_pos"][:] = o2_pos
        group["coulomb_interaction"][:] = coulomb_interaction
