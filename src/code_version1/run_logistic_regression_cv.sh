#!/bin/bash
#SBATCH -A berzelius-2025-214 
#SBATCH --job-name=logreg
#SBATCH --output= # suitable output path
#SBATCH --error= # suitable error path
#SBATCH -c 1
#SBATCH -p berzelius-cpu
#SBATCH -t 10:00:00
##SBATCH --array=1-600%33 # useful when testing large combination spaces

if [ -z $1 ]
then
        offset=0
else
        offset=$1
fi
i=$(( SLURM_ARRAY_TASK_ID + offset ))

# Load necessary modules
module load Mambaforge/23.3.1-1-hpc1-bdist # replace with suitable module for your cluster/system
mamba activate # suitable environment

# Set script variables
SCRIPT_PATH= # path to script, include python script as well

# Run the Python script with arguments

# test exactly one feature from each cluster
python3 $SCRIPT_PATH --min-k 1 --max-k 1 --optimize-metric tpr_at_xfpr --target-fpr 0.01 --fpr-tol 0.005 --use-tqdm # recommend not using tqdm when submitting to cluster

# example flags 
#--each-feature-1-cluster #--split-combos --n-splits-combos 600 --current-combo $i #--use-tqdm #--no-fseek 