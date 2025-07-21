#!/bin/bash
# Always run this script from the root of the deliverable2
make converter || exit;

echo "CONVERTING Will take a huge amount of time (up to 10 min)"
./build/mtx_to_sbmtx $1