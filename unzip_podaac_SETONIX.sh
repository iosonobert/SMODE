# Activate your virtual environment
source ~/smode_env/bin/activate

# Make sure download folder exists
mkdir -p $MYSCRATCH/DOPPVISs

# Optional: extract downloaded .gz files
for f in $MYSCRATCH/DOPPVIS/*.gz; do
    echo "Extracting $f"
    # tar -xzvf "$f" -C $MYSCRATCH/DOPPVIS
    first_entry=$(tar -tzf "$f" | head -1 | cut -d/ -f1)
    dest="$MYSCRATCH/DOPPVIS/$first_entry"

    if [ ! -d "$dest" ]; then
        tar -xzvf "$f" -C "$MYSCRATCH/DOPPVIS"
    else
        echo "Archive $dest already extracted, not extracting again."
    fi
done