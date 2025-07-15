#!/bin/bash
# Always run this script from the root of the repo

# COMPILE
echo -e "\nCOMPILING"
module load CUDA/12.5 || exit;
make normal || exit;

# Pre run
mkdir -p output || exit;
rm -f output/result.err output/result.out output/finished.out output/partial.csv;

# QUEUE
echo -e "\nQUEUED JOB (may take some time to be started)"
sbatch gpu-run.sbatch $1 >/dev/null || exit;

# WAIT JOB START
until [ -e ./output/result.err ]; do printf .; sleep 1; done;

# WAIT JOB FINISH
echo -e "\nJOB STARTED REMOTELY, may take very long to finish (depending on matrix size)"
until [ -e ./output/finished.out ]; do printf .; sleep 1; done;
echo;

