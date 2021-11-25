#!/bin/bash

#SBATCH --partition=microcloud
#SBATCH --ntasks=1              # Requested (MPI)tasks. Default=1
#SBATCH --cpus-per-task=1       # Requested CPUs per task. Default=1
#SBATCH --mem=24G               # Memory limit. [1-999][K|M|G|T]
#SBATCH --time=24:00:00            # Time limit. [[days-]hh:]mm[:ss]

### configure file to store console output.
### Write output to /dev/null to discard output
#SBATCH --output=electronegs.log

### give your job a name (and maybe a comment) to find it in the queue
#SBATCH --job-name=100_molecules_data_prep
#SBATCH --comment="gpu example script"

### load environment modules
module purge

### set environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source use_conda_env.sh

srun python hdf5_to_geometric.py > hdf5_to_geometric.out
echo "STEP 1 DONE"
srun python dataset_from_hdf5.py > dataset_from_hdf5.out
echo "STEP 2 DONE"
