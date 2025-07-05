#!/bin/bash
echo 'COMPILING'
make profiling || exit;

echo 'PROFILING (may take some time to finish)';
sudo $(which ncu) --config-file off --export report.ncu-rep --force-overwrite --set full build/main_profiling $1