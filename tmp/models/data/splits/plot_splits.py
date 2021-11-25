import matplotlib.pyplot as plt

exec(open("dataset_class.py").read())

training_dataset = torch.load("train_dset.pt")
testing_dataset = torch.load("test_dset.pt")
validation_dataset = torch.load("val_dset.pt")


num_atoms_training = []
num_atoms_testing = []
num_atoms_validation = []


for data in training_dataset:
    num_atoms_training.append(data.x.shape[0] - 2)

for data in testing_dataset:
    num_atoms_testing.append(data.x.shape[0] - 2)

for data in validation_dataset:
    num_atoms_validation.append(data.x.shape[0] - 2)

assert len(num_atoms_training) == len(training_dataset), "unexpected error in training dataset"
assert len(num_atoms_testing) == len(testing_dataset), "unexpected error in testing dataset"
assert len(num_atoms_validation) == len(validation_dataset), "unexpected error in validation dataset"

plt.hist(num_atoms_training)
plt.title("training dataset")
plt.savefig("training_dataset_histogram.png")
plt.clf()

plt.hist(num_atoms_validation)
plt.title("validation dataset")
plt.savefig("validation_dataset_histogram.png")
plt.clf()

plt.hist(num_atoms_testing)
plt.title("testing dataset")
plt.savefig("testing_dataset_histogram.png")
