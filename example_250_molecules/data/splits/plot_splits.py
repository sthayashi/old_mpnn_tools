import matplotlib.pyplot as plt
import numpy as np

import time

exec(open("dataset_class.py").read())

# Returns dict of {number_of_atoms : num_points}
def count_sizes(dataset):
    molecule_sizes = {}
    for data in dataset:
        num_atoms = data.x.shape[0]-2
        if num_atoms not in molecule_sizes.keys():
            molecule_sizes[num_atoms] = 1
        else:
            molecule_sizes[num_atoms] += 1
    return molecule_sizes
    
##########################################################################
training_dataset = torch.load("train_dset.pt")
testing_dataset = torch.load("test_dset.pt")
validation_dataset = torch.load("val_dset.pt")

num_atoms_training = []
num_atoms_testing = []
num_atoms_validation = []

training_y = []
testing_y = []
validation_y = []

for data in training_dataset:
    num_atoms_training.append(data.x.shape[0] - 2)
    training_y.append(data.y)

for data in testing_dataset:
    num_atoms_testing.append(data.x.shape[0] - 2)
    testing_y.append(data.y)

for data in validation_dataset:
    num_atoms_validation.append(data.x.shape[0] - 2)
    validation_y.append(data.y)

assert len(num_atoms_training) == len(training_dataset), "unexpected error in training dataset"
assert len(num_atoms_testing) == len(testing_dataset), "unexpected error in testing dataset"
assert len(num_atoms_validation) == len(validation_dataset), "unexpected error in validation dataset"

#### distributions of datasets

plt.hist(num_atoms_training)
plt.title("training dataset")
plt.xlabel("Number of Atoms")
plt.savefig("training_dataset_histogram.png")
time.sleep(0.1)
plt.clf()

plt.hist(num_atoms_validation)
plt.title("validation dataset")
plt.xlabel("Number of Atoms")
plt.savefig("validation_dataset_histogram.png")
time.sleep(0.1)
plt.clf()

plt.hist(num_atoms_testing)
plt.title("testing dataset")
plt.xlabel("Number of Atoms")
plt.savefig("testing_dataset_histogram.png")
time.sleep(0.1)
plt.clf()

#### transfer integrals

plt.hist(training_y)
plt.title("training dataset")
plt.xlabel("TI (truth)")
plt.savefig("training_ti_histogram.png")
time.sleep(0.1)
plt.clf()

plt.hist(testing_y)
plt.title("testing dataset")
plt.xlabel("TI (truth)")
plt.savefig("testing_ti_histogram.png")
time.sleep(0.1)
plt.clf()

plt.hist(validation_y)
plt.title("validation dataset")
plt.xlabel("TI (truth)")
plt.savefig("validation_ti_histogram.png")
time.sleep(0.1)

plt.close()
