# Activate your virtual environment
source ~/smode_env/bin/activate

# Make sure download folder exists
mkdir -p $MYSCRATCH/DOPPVISs

# Optional: extract downloaded .gz files
for f in $MYSCRATCH/DOPPVIS/*.gz; do
    echo "Extracting $f"
    tar -xzvf "$f" -C $MYSCRATCH/DOPPVIS
done