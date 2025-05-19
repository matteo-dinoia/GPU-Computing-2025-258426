# COMPILE
echo 'COMPILING'
module load CUDA/12.5 || exit;
make all || exit;

# RUN
echo 'QUEUEING JOB'
rm -f result.err;
rm -f finished.out;
sbatch gpu-run.sbatch $1 >/dev/null || exit;
echo 'QUEUED JOB (may take some time to be started)'

# WAIT JOB START
until [ -e ./result.err ]; do printf .; sleep 1; done;
echo

# WAIT JOB FINISH
echo 'JOB STARTED (may take some time to finish)';
until [ -e ./finished.out ]; do printf .; sleep 1; done;
echo

# PRINT RESULTS
cat ./result.err;
echo ###################################################
cat ./result.out