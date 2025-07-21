#!/usr/bin/bash
# Always run this script from the root of the deliverable2
# require "rsync"

# change to your own if any is needed
SSH_FLAG="-i ~/.ssh/cluster"
# change to own
REMOTE="matteo.dinoia@baldo.disi.unitn.it"

DEST="matteo.dinoia_deliverable2"
INTERNAL_RUNNER="scripts/gpu-internal-run.sh"
TO_COPY=" Makefile src distributed_mmio datasets scripts"

# TEST MAKING LOCALLY BEFORE PUSHING
echo "TESTING THE PROGRAM COMPILE LOCALLY"
make normal || exit;

# EXECUTE ON CLUSTER
echo -e "\nCOPYING FILES"
rsync -e "ssh $SSH_FLAG" -aup --delete --progress ${TO_COPY} ${REMOTE}:${DEST}/  || exit;

# EXECUTE IT
COMMAND="cd ${DEST}/ && ./${INTERNAL_RUNNER} $@"
ssh ${SSH_FLAG} matteo.dinoia@baldo.disi.unitn.it ${COMMAND}

# COPY RESULTS
rm -fr output/tmp/*
scp -r ${SSH_FLAG} ${REMOTE}:${DEST}/output/tmp/* output/tmp/ > /dev/null
echo -e "\nResulting output file can be found at 'output/tmp/'"
