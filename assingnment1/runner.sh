#!/usr/bin/bash
module load CUDA/12.5 || exit;

make all || exit;

rm -f result.err
rm -f result.out

sbatch gpu-run.sbatch || exit;

#echo "WAITING: Put in queue."
#sleep 5;
until [ -e ./result.err ]; do sleep 1; done
until [ -e ./result.out ]; do sleep 1; done
sleep 1;

cat ./result.err
echo "###################################################"
cat ./result.out

