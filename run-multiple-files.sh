#!/bin/sh
# This Script was largely inspired by the script provided by Beatrice Alex, who wrote the initial version, which wasn't parallelised yet
# Authors: Beatrice Alex & Sebastian Gmür
# Date: 2025-01-10

usage="./run-multiple-files -i inputDirectory -o outputDirectory"

# Export the required environment variable
export GEOPARSER_DB_COMMAND="/Users/sebastiangmur/Projekte/UZH_HS24/GIR/edinburg_geoparser/new_install_dir/bin/lxpostgresql -u postgres -p password -d geonames"

# Check that some options are specified
if [ $# -eq "0" ]; then
    echo "No arguments specified"
    echo "usage: $usage" >&2
    exit 2
fi

# Parse the command-line arguments
while test $# -gt 0; do
    arg=$1
    shift
    case $arg in
    -i)
        inputdirname=$1
        shift 1
        ;;
    -o)
        outputdirname=$1
        shift 1
        ;;
    *)
        echo "Wrong argument specified"
        echo "usage: $usage" >&2
        exit 2
        ;;
    esac
done

# Ensure input and output directories are provided
if [ -z "$inputdirname" ] || [ -z "$outputdirname" ]; then
    echo "Both input and output directories must be specified."
    echo "usage: $usage" >&2
    exit 2
fi

# Get the total number of files to process
total_files=$(find "$inputdirname" -maxdepth 1 -name "*.txt" | wc -l)
start_time=$SECONDS

# Function to process each file
process_file() {
    file=$1
    outputdirname=$2
    prefix=$(basename "$file" ".txt")
    output_file="$outputdirname/$prefix.out.xml"
    
    # Check if the output file already exists
    if [ -f "$output_file" ]; then
        echo "Skipping $file -> $output_file (already exists)"
        return
    fi
    
    echo "Processing $file -> $output_file"
    ./run -t plain -g geonames-local < "$file" > "$output_file"
}

export -f process_file

# Use find and xargs for parallel processing
find "$inputdirname" -maxdepth 1 -name "*.txt" | \
    xargs -P 4 -I {} bash -c 'process_file "$@"' _ {} "$outputdirname"

# Calculate and print elapsed time
elapsed_time=$((SECONDS - start_time))
echo "Finished processing $total_files files in $elapsed_time seconds."