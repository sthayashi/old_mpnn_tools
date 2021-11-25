This repo is old code that was used to generate the dataset for a graph neural network project. Uses data published in Nature by [Stuke, A., Kunkel, C., Golze, D. et al.](https://rdcu.be/cB3B2). Due to old packages which only exist in python2, the code in this repo is meant to be version agnostic (e.g. no type hints).

The project's purpose was for predicting charge transfer integrals on organic molecules.

## The behavior of each script is as follows:
- `parse_df.py` takes a json input from the Oberhofer et. al. dataset, and collect necessary data for calculating transfer integrals and coulomb interactions. 
                Outputs a h5 file that is the input for `generate_o2_pos.py`.
- `parse_df-uniform.py` does the same thing as `parse_df.py`, but it uniformly samples molecules according to number of atoms. E.g. molecules with {4, 5, 6} atoms, 5 at each size.
- `generate_o2_pos.py` generates o2 molecule positions around molecule(s).
- `py2_dipro.py` calculates the transfer integrals.
- `hdf5_to_geometric.py` generates node and edge features from the previous steps into an easily read form for pytorch, namely node and edge features. See also `add_features_hdf5.jl`.
- `dataset_from_hdf5.py` takes the output of `hdf5_to_geometric.py` and saves it as a pytorch geometric dataset. Note the terrible naming of this and the previous file.
- `template` directory that functions as a blank slate for training a model.

## Workflow from start to finish:
- `parse_df.py` takes path to the dataset and sampling batches of organic molecules. Outputs `parsed_molecules.h5`.
- `generate_o2_pos.py` generates oxygen molecules around each organic molecule using a Sobol sampler. Appends to `parsed_molecules.h5`.
- `py2_dipro.py` calculates the transfer integrals.
- Move `parsed_moecules.h5` to `template/preprocessing/` and run `hdf5_to_geometric.py`.
- Run `dataset_from_hdf5.py`. The pyg dataset should now be in `template/data/processed/`.

## General remarks to make reading code easier:
- There is a field stored called `good_o2_inds`. This refers to the transfer integral calculations which didn't fail.
- The rationale of data structures can be described as:
    - For an array shape (MxNxK), the axes M refers to the different organic + O2 molecule pairs.
    - For properties which can be reused, e.g. the organic molecule coordinates, we simply have an (number_of_atoms x 3) array.
    - Similarly, tranfer integrals are simply shape (M,).

## As an example of the workflow, see the directory `example_250_molecules`
1. `example_250_molecules/parsed_250_molecules/`: run the slurm script `run_workflow.sh`. This will do all steps up to and including transfer integrals.
2. Copy `example_25_molecules/parsed_250_molecules/parsed_molecules.h5` to `example_250_molecules/preprocess` and run
    - `hdf5_to_geometric.py`
    - `dataset_from_hdf5.py`
3. From step 2., there should now be a file `example_250_molecules/data/processed/processed_data.dataset`. This is the pytorch specific data object that is used by pytorch geometric.
4. Train the model using `example_250_molecules/model/train_gnn_test.sh`

Packages needed:

- `h5py`
- `pandas`
- `numpy`
- `mendeleev`
- `pyquaternion`
- `salib`
- `scipy`

Additional QOL packages:
- `tqdm` for progress bars
