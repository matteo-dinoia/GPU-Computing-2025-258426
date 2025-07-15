#!/bin/bash
make converter || exit;

echo "CONVERTING Will take a huge amount of time (up to 10 min)"
./build/mtx_to_sbmtx $1