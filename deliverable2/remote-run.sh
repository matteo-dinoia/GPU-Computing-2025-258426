#!/usr/bin/bash
# require "rsync"

DEST="deliverable2/"
INTERNAL_RUNNER="gpu-internal-run.sh"
TO_COPY="${INTERNAL_RUNNER} gpu-run.sbatch Makefile src distributed_mmio MtxMan datasets profile.sh"

# EXECUTE ON CLUSTER
echo "COPYING FILES"
rsync -e 'ssh -i ~/.ssh/cluster' -aup --delete --progress ${TO_COPY} matteo.dinoia@baldo.disi.unitn.it:${DEST}  || exit;

# EXECUTE IT
COMMAND="cd ${DEST} && ./${INTERNAL_RUNNER} $1"
ssh -i ~/.ssh/cluster matteo.dinoia@baldo.disi.unitn.it ${COMMAND}
