#!/usr/bin/bash
# require "rsync"

# change to your own if any is needed
SSH_FLAG="-i ~/.ssh/cluster"
# change to own
REMOTE="matteo.dinoia@baldo.disi.unitn.it"
# change name to free folder you can use
DEST="deliverable2"

INTERNAL_RUNNER="gpu-internal-run.sh"
TO_COPY="${INTERNAL_RUNNER} gpu-run.sbatch Makefile src distributed_mmio datasets profile.sh"

# TEST MAKING LOCALLY BEFORE PUSHING
echo "TESTING THE PROGRAM COMPILE LOCALLY"
make normal || exit;

# EXECUTE ON CLUSTER
echo -e "\nCOPYING FILES"
rsync -e "ssh $SSH_FLAG" -aup --delete --progress ${TO_COPY} ${REMOTE}:${DEST}/  || exit;

# EXECUTE IT
COMMAND="cd ${DEST}/ && ./${INTERNAL_RUNNER} $1"
ssh ${SSH_FLAG} matteo.dinoia@baldo.disi.unitn.it ${COMMAND}

# COPY RESULTS
scp ${SSH_FLAG} ${REMOTE}:${DEST}/output/result.out output/result.out
scp ${SSH_FLAG} ${REMOTE}:${DEST}/output/result.err output/result.err
scp ${SSH_FLAG} ${REMOTE}:${DEST}/output/partial.csv output/partial.csv

# PRINT RESULTS
cat ./output/result.err;
echo ###################################################
cat ./output/result.out
