import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File("parsed_molecules.h5", 'r')
refcodes = list(f.keys())

smallest_set = np.inf
number_of_atoms = []
dataset_size = []

# Find smallest dataset size
for refcode in refcodes:
    molecule_data = f.get(refcode)

    organic_pos = molecule_data.get("molecule_pos")[:]

    inds = molecule_data.get("o2_good_inds")[:]
    num_good_samples = len(inds)

    if num_good_samples < smallest_set:
        smallest_set = num_good_samples

    number_of_atoms.append(organic_pos.shape[0]-2)
    dataset_size.append(num_good_samples)
f.close()

plt.hist(dataset_size)
plt.title("Size of Dataset per Organic Molecule")
plt.xlabel("Number of Datapoints")
plt.savefig("dataset_distribution.png")
plt.close()
