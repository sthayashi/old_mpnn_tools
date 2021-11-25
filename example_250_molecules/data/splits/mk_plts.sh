#!/bin/bash

#SBATCH --partition=microcloud
#SBATCH --ntasks=1              # Requested (MPI)tasks. Default=1
#SBATCH --cpus-per-task=4       # Requested CPUs per task. Default=1
#SBATCH --mem=36G               # Memory limit. [1-999][K|M|G|T]
#SBATCH --time=00:30:00            # Time limit. [[days-]hh:]mm[:ss]

### configure file to store console output.
### Write output to /dev/null to discard output
#SBATCH --output=results.log

### configure email notifications
### mail types: BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80
#SBATCH --mail-user=scott.hayashi@tum.de
##SBATCH --mail-type=END,FAIL,TIME_LIMIT

### give your job a name (and maybe a comment) to find it in the queue
#SBATCH --job-name=plots

### load environment modules
module purge

### set environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

### Run dipro with pdynamo env
source use_conda_env.sh

srun python plot_splits.py > results.out
