# Activate your virtual environment
source ~/smode_env/bin/activate

INSTRUMENT="$1"
if [ -z "$INSTRUMENT" ]; then
    INSTRUMENT=LWIR
fi
echo "Instrument: $INSTRUMENT"

# Make sure download folder exists
mkdir -p $MYSCRATCH/$INSTRUMENT

# Optional: extract downloaded .gz files
for f in $MYSCRATCH/$INSTRUMENT/*.gz; do
    echo "Extracting $f"
    # tar -xzvf "$f" -C $MYSCRATCH/$INSTRUMENT
    first_entry=$(tar -tzf "$f" | head -1 | cut -d/ -f1)
    dest="$MYSCRATCH/$INSTRUMENT/$first_entry"

    if [ ! -d "$dest" ]; then
        tar -xzvf "$f" -C "$MYSCRATCH/$INSTRUMENT"
    else
        echo "Archive $dest already extracted, not extracting again."
    fi
done