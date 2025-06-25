#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --partition=edu-short
#SBATCH --nodes=1

#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

./build/main $1
touch finished.out
