#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --nodelist=edu02
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#for matrix in $(ls ../matteo.dinoia_deliverable2/datasets/*.*); do echo ${matrix}; ./build/main-cpu ${matrix}; done
./build/main-cpu $1
