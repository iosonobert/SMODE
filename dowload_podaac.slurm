#!/bin/bash
#SBATCH --job-name=smode_download
#SBATCH --output=$HOME/SMODE/smode_download.out
#SBATCH --error=$HOME/SMODE/smode_download.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --partition=copy

# Activate virtual environment
source $HOME/smode_env/bin/activate

# Run download script
bash $HOME/SMODE/download_smode_login.sh