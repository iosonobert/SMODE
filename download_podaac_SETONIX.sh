# Activate your virtual environment
source $MYSOFTWARE/smode_env/bin/activate

# Make sure download folder exists
mkdir -p $MYSCRATCH/DOPPVIS

# Usage: ./download_script.sh 2023-04-19
DATE="$1"
STARTTIME="$2"
ENDTIME="$3"

if [ -z "$DATE" ]; then
    echo "Usage: $0 YYYY-MM-DD"
    exit 1
fi
if [ -z "$STARTTIME" ]; then
    STARTTIME=00:00:00
fi
if [ -z "$ENDTIME" ]; then
    ENDTIME=23:59:59
fi

echo "Downloading data for date: $DATE from $STARTTIME to $ENDTIME"

# Run the download
podaac-data-downloader \
  -c SMODE_L1_MASS_DOPPVIS_V1 \
  -d $MYSCRATCH/DOPPVIS \
  --start-date ${DATE}T${STARTTIME}Z \
  --end-date ${DATE}T${ENDTIME}Z \
  -e ""

echo "Download complete."

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