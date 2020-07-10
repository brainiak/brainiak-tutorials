#!/bin/bash -i
#SBATCH -t 20
#SBATCH --mem-per-cpu=8G
#SBATCH -n 2
#SBATCH --job-name fcma_voxel_select_cv
#SBATCH --output=../logs/fcma_voxel_select_cv-%j.out

# Set up the environment. You will need to modify the module for your cluster.
source ../setup_environment.sh

# How many threads can you make
export OMP_NUM_THREADS=32

# set the current dir
currentdir=`pwd`

# Prepare inputs to voxel selection function
data_dir=$1  # What is the directory containing data?
suffix=$2  # What is the extension of the data you're loading
mask_file=$3  # What is the path to the whole brain mask
epoch_file=$4  # What is the path to the epoch file
left_out_subj=$5  # Which participant (as an integer) are you leaving out for this cv?
output_dir=$6 # Where do you want to save the data

# Run the script
if [ $configuration == "cluster" ]
then
	srun --mpi=pmi2 python ./fcma_voxel_selection_cv.py $data_dir $suffix $mask_file $epoch_file $left_out_subj $output_dir
else
	mpirun -np 2 python ./fcma_voxel_selection_cv.py $data_dir $suffix $mask_file $epoch_file $left_out_subj $output_dir
fi
