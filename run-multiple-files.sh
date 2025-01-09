#!/bin/sh
# Author: Beatrice Alex
# Date: 28-01-2016
# Description: Run the Geoparser on multiple text files in parallel using xargs

usage="./run-multiple-files -i inputDirectory -o outputDirectory"

if [ $# -eq "0" ]; then
    echo "No arguments specified"
    echo "usage: $usage" >&2
    exit 2
fi

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

if [ -z "$inputdirname" ] || [ -z "$outputdirname" ]; then
    echo "Both input and output directories must be specified."
    echo "usage: $usage" >&2
    exit 2
fi

total_files=$(find "$inputdirname" -maxdepth 1 -name "*.txt" | wc -l)
start_time=$SECONDS

process_file() {
    file=$1
    prefix=$(basename "$file" ".txt")
    echo "Processing $file"
    ./run -t plain -g geonames -o "$outputdirname" "$prefix" < "$file"
}

export -f process_file
export outputdirname

find "$inputdirname" -maxdepth 1 -name "*.txt" -print0 | \
    xargs -0 -P 4 -I {} bash -c 'process_file "$@"' _ "{}"

elapsed_time=$((SECONDS - start_time))
echo "Finished processing $total_files files in $elapsed_time seconds."