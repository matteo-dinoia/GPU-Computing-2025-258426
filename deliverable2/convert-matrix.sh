#!/bin/bash
# Always run this script from the root of the repo

make converter || exit;

echo "CONVERTING"
./build/mtx_to_bmtx $1