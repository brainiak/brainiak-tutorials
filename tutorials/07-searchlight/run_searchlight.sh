#!/bin/bash -i
# Input python command to be submitted as a job

#SBATCH --output=../logs/searchlight-%j.out
#SBATCH --job-name searchlight
#SBATCH -t 30        # time limit: how many minutes 
#SBATCH --mem=4G        # memory limit
#SBATCH -n 2         # how many cores to use

# Set up the environment
source ../setup_environment.sh

# Run the python script (use mpi if running on the cluster)
if [ $configuration == "cluster" ]
then
	srun --mpi=pmi2 python ./searchlight.py
else
	python ./searchlight.py
fi
