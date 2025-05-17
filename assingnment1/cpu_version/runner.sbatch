#!/bin/bash

#SBATCH --job-name=matteo_job_0
#SBATCH --output=my_output_%j.out
#SBATCH --error=my_error_%j.err
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

./main.out
