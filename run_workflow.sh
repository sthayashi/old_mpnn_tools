#!/bin/bash

#SBATCH --partition=carlos
#SBATCH --ntasks=1              # Requested (MPI)tasks. Default=1
#SBATCH --cpus-per-task=1       # Requested CPUs per task. Default=1
#SBATCH --mem=48G               # Memory limit. [1-999][K|M|G|T]
#SBATCH --time=4:30:00            # Time limit. [[days-]hh:]mm[:ss]

### configure file to store console output.
### Write output to /dev/null to discard output
#SBATCH --output=py2_dipro.log

### configure email notifications
### mail types: BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80
##SBATCH --mail-user=your.mail@tum.de
##SBATCH --mail-type=END,FAIL,TIME_LIMIT

### give your job a name (and maybe a comment) to find it in the queue
#SBATCH --job-name=tune_bayesopt
#SBATCH --comment="gpu example script"

### load environment modules
module purge

### set environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

### Run parsing scripts with python3 env
source use_py3_env.sh

srun python parse_df.py
srun python generate_o2_pos.py

### Run dipro with pdynamo env
source use_pdynamo_env.sh

srun python py2_dipro.py > py2_dipro.out
