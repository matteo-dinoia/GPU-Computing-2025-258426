#!/bin/bash
# Always run this script from the root of the deliverable2

make converter || exit;

echo "CONVERTING"
./build/mtx_to_bmtx $1