# Activate your virtual environment
source ~/smode_env/bin/activate

# Make sure download folder exists
mkdir -p ./DOPPVIS

# Run the download
podaac-data-subscriber \
  -c SMODE_L1_MASS_DOPPVIS_V1 \
  -d ./DOPPVIS \
  --start-date 2023-04-19T00:00:00Z \
  --end-date 2023-04-20T00:00:00Z \
  -e ""

# Optional: extract downloaded .gz files
for f in ./DOPPVIS/*.gz; do
    tar -xzvf "$f" -C ./DOPPVIS
done