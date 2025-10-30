#!/bin/bash
#SBATCH --job-name=smode_download
#SBATCH --output=smode_download.out
#SBATCH --error=smode_download.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --partition=copy

# Print job ID into the output file
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "User inputs for date and time: $1 $2 $3"

# Activate virtual environment
source $MYSOFTWARE/smode_env/bin/activate

# Run download script
bash $MYSOFTWARE/code/SMODE/download_podaac_SETONIX.sh $1 $2 $3
