#!/bin/bash -i

# Specify the code necessary to setup your environment to run BrainIAK on a Jupyter notebook. This could involve activating a conda environment (like below) or importing modules.
CONDA_ENV=mybrainiak

# How are you interacting with the notebooks? On a cluster, locally on a laptop, using docker, etc.? This will determine how some functions are launched, such as jupyter and some jobs
configuration='server' # includes 'cluster' or 'local' or 'docker'

# Also setup the environment to use some simple visualization tools, like FSL
#module load FSL

# If on a cluster, specify the server name you are going to use. This might be the address you use for your SSH key to log in to the cluster. The default is to use the host name that may be appropriate
server=$(hostname)

if [[ -n $CONDA_ENV ]]; then
    # Start the conda environment
    conda --version  &> /dev/null
    if [[ $? -eq 0 ]]; then
        # conda command is present
        conda activate $CONDA_ENV
    else
        # older versions of conda use source activate instead
        source activate $CONDA_ENV
    fi

    # Check if the conda command succeeded
    if [[ $? -ne 0 ]]; then
        echo "Conda not initialized properly, check your conda environment"
        exit -1
    fi
fi

