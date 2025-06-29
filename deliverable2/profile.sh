#!/bin/bash
echo 'COMPILING'
make normal || exit;

echo 'PROFILING (may take some time to finish)';
sudo $(which ncu) --config-file off --export report.ncu-rep --force-overwrite --set full build/main $1