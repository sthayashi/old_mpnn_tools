#!/bin/bash

#SBATCH --partition=gpucloud
#SBATCH --ntasks=1              # Requested (MPI)tasks. Default=1
#SBATCH --cpus-per-task=4       # Requested CPUs per task. Default=1
#SBATCH --mem=64G               # Memory limit. [1-999][K|M|G|T]
#SBATCH --time=72:00:00            # Time limit. [[days-]hh:]mm[:ss]
#SBATCH --gpus=1

### configure file to store console output.
### Write output to /dev/null to discard output
#SBATCH --output=results.log

### configure email notifications
### mail types: BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80
#SBATCH --mail-user=scott.hayashi@tum.de
##SBATCH --mail-type=END,FAIL,TIME_LIMIT

### give your job a name (and maybe a comment) to find it in the queue
#SBATCH --job-name=TI_100_molecules
#SBATCH --comment="transfer integrals for 100 molecules"

### load environment modules
module purge

### set environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

### Run dipro with pdynamo env
source use_conda_env.sh

srun python 3-run.py > results.out
