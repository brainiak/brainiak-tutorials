#!/bin/bash -i
#SBATCH -t 15
#SBATCH --mem-per-cpu=12G
#SBATCH -n 2
#SBATCH --job-name fcma_classify
#SBATCH --output=../logs/fcma_classify-%j.out

# Set up the environment. You will need to modify the module for your cluster.
source ../setup_environment.sh

# How many threads can you make
export OMP_NUM_THREADS=32

# set the current dir
currentdir=`pwd`

# Prepare inputs to voxel selection function
data_dir=$1  # What is the directory containing data?
suffix=$2  # What is the extension of the data you're loading 
mask_file=$3  #What is the path to the top N mask file (THIS IS NOT THE WHOLE BRAIN MASK)
epoch_file=$4  # What is the path to the epoch file
left_out_subj=$5  #Which participant (as an integer) are you using for testing? 
second_mask=$6  # Do you want to use a second mask to compare the data with? Necessary for extrinsic analyses. Otherwise ignore this input or set to None

# Run the script
# Run the python script (use mpi if running on the cluster)
if [ $configuration == "cluster" ]
then
	srun --mpi=pmi2 python ./fcma_classify.py $data_dir $suffix $mask_file $epoch_file $left_out_subj $second_mask
else
	echo Cannot run python ./fcma_classify.py $data_dir $suffix $mask_file $epoch_file $left_out_subj $second_mask
	echo Must have two cores in order to setup master and worker control
	echo Try running this on a cluster
fi
