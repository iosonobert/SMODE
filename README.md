# SMODE

## Netrc

You'll need to scp your .netrc file onto your HPC home directory. Assuming you've done this locally on your local linux machine. 

`
scp ~/.netrc azulberti@setonix.pawsey.org.au:~/.netrc
`

## EXIFTOOL 

Next you'll need Image-ExifTool-12.69. Again assuming you've got this locally on your local linux machine. 

`
scp -r ~/Image-ExifTool-12.69 azulberti@setonix.pawsey.org.au:<YOUR_SOFTWARE_FOLDER>
`

And then change the permissions.

`
chmod +x $MYSOFTWARE$/Image-ExifTool-12.69/exiftool
`

## Downloading data

Data downloads using 

`
sbatch download_podaac.slurm YYYY-MM-DD
`

Which does a full 24 h by default, or:

`
sbatch download_podaac.slurm YYYY-MM-DD 10:00:00 10:00:05
`

if you sish to subset. 

By default this runs 30 H jobs on the COPY partition. This is probably good for a full day including extraction.  

## LWIR and DOPPVIS

These 2 different instruments can be downloaded by the fourth input to download_podaac.slurm

`
sbatch download_podaac.slurm YYYY-MM-DD 10:00:00 10:00:05 LWIR
sbatch download_podaac.slurm YYYY-MM-DD 10:00:00 10:00:05 DOPPVIS
`

Pretty annoying that I made it the fourth input. I should change that.  

## Unzipping data

The download script unzips, but if for whatever reason it doesn't finish use

`
sbatch unzip_podaac.slurm LWIR
sbatch unzip_podaac.slurm DOPPVIS
`

By default this runs 4 H jobs on the COPY partition and only unxips folders that haven't been partially unzipped. Patials will be a big gotya. 

## Extracting EXIF data

As you might expect:

By default this runs 4 H jobs on the COPY partition 

`
sbatch extract_exif.slurm
`

This makes netcdfs for all unzipped folders that don't yet have netcdfs. 
