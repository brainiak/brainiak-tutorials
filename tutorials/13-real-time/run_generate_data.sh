#!/bin/bash -i
# Input python command to be submitted as a job

#SBATCH --output=../generate_data-%j.out
#SBATCH --job-name generate_data
#SBATCH -t 30
#SBATCH -m=4G
#SBATCH -n 1

MY_DIR=$(dirname "$0")

# Set up the environment
source $MY_DIR/../setup_environment.sh

# Run the python script
python $MY_DIR/generate_data.py
