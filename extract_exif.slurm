#!/bin/bash
#SBATCH --job-name=smode_exif
#SBATCH --output=smode_exif.out
#SBATCH --error=smode_exif.out
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --partition=work

# Print job ID into the output file
echo "SLURM Job ID: $SLURM_JOB_ID"

INSTRUMENT="$1"
echo "User input for instrument: $INSTRUMENT"
if [ -z "$INSTRUMENT" ]; then
    echo ERROR: Instrument argument is required.
    echo "Usage: $0 INSTRUMENT"
    exit 1
fi

module load python/3.11.6

# Activate virtual environment
source $MYSOFTWARE/smode_env/bin/activate

# Run download script
python3 -u extract_exif.py "$INSTRUMENT"