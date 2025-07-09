#!/bin/bash
make converter || exit;

./build/mtx_to_bmtx $1