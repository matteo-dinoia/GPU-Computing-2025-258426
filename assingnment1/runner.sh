#!/usr/bin/bash
# require "rsync"

# possible datasets
# datasets/circuit5M_dc.mtx datasets/mawi_201512020330.mtx datasets/mycielskian3.mtx

DEST="assignment1/"
TO_COPY="int-runner.sh Makefile gpu-run.sbatch src"

# EXECUTE ON CLUSTER
echo "COPYING FILES"
rsync -e 'ssh -i ~/.ssh/cluster' -aup --delete $TO_COPY src matteo.dinoia@baldo.disi.unitn.it:$DEST

# EXECUTE IT
COMMAND="cd assignment1 && ./int-runner.sh $1"
ssh -i ~/.ssh/cluster matteo.dinoia@baldo.disi.unitn.it $COMMAND
