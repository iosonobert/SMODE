#!/bin/bash
#SBATCH --job-name=smode_download
#SBATCH --output=smode_download.out
#SBATCH --error=smode_download.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --partition=copy

# Print job ID into the output file
echo "SLURM Job ID: $SLURM_JOB_ID"

module load python/3.11.6

# Activate virtual environment
source $HOME/smode_env/bin/activate

# Run download script
python3 extract_exif.py