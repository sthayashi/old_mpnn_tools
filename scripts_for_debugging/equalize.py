import h5py
import numpy as np

f = h5py.File("parsed_molecules.h5", 'r')
refcodes = list(f.keys())
smallest_set = np.inf

# Find smallest dataset size
for refcode in refcodes:
    molecule_data = f.get(refcode)

    inds = molecule_data.get("o2_good_inds")[:]
    num_good_samples = len(inds)

    if num_good_samples < smallest_set:
        smallest_set = num_good_samples
f.close()

# Randomly sample each molecule dataset using the aforementioned size
f = h5py.File("parsed_molecules.h5", 'a')
for refcode in refcodes:
    molecule_data = f.get(refcode)

    inds = molecule_data.get("o2_good_inds")[:]
    sampled_inds = np.sort(np.random.choice(inds, smallest_set, replace=False))

    molecule_data.require_dataset("sampled_inds", shape=sampled_inds.shape, dtype=sampled_inds.dtype)
    molecule_data["sampled_inds"][:] = sampled_inds
f.close()
