#!/bin/bash
# Input python command to be submitted as a job

#SBATCH --output=generate_data-%j.out
#SBATCH --job-name generate_data
#SBATCH -t 30
#SBATCH --mem=4G
#SBATCH -n 1

# Check you are in the correct directory
if [ ${PWD##*/} == '13-real-time' ]
then
	cd ..
	echo "Changing to the tutorials directory"
fi


# Set up the environment
source ./setup_environment.sh

# Run the python script
python ./13-real-time/generate_data.py
