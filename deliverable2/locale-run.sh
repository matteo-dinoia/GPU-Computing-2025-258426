#!/usr/bin/bash
# Always run this script from the root of the repo

echo "COMPILING";
make normal || exit;

# Pre run
mkdir -p output || exit;
rm -f output/result.err output/result.out output/partial.csv;

echo -e "\nRUNNING LOCALLY, may take very long to finish (depending on matrix size)";
./build/main $1;