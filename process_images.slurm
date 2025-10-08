#!/bin/bash
#SBATCH --job-name=smode_process
#SBATCH --output=smode_process.out
#SBATCH --error=smode_process.out
#SBATCH --time=01:00:00
#SBATCH --mem=230G
#SBATCH --partition=work

# Print job ID into the output file
echo "SLURM Job ID: $SLURM_JOB_ID"

module load python/3.11.6

# Activate virtual environment
source $MYSOFTWARE/smode_env/bin/activate

# Run download script
python3 -u process_images.py