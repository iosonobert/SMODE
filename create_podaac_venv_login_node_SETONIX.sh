
# Load Python module if needed
module load python/3.11.6   # replace with the correct version

# Go to your home directory or project folder
cd $HOME

# Create virtual environment (one-time)
python3 -m venv smode_env

# Activate it
source smode_env/bin/activate

# Upgrade pip and install packages
pip install --upgrade pip
pip install podaac-data-subscriber

# Deactivate when done
deactivate