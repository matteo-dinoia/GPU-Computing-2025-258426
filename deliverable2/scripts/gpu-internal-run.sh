#!/bin/bash
# Always run this script from the root of the deliverable2

NODE="edu01"


# COMPILE
echo -e "\nCOMPILING"
module load CUDA/12.5 || exit;
make normal || exit;

# Pre run
mkdir -p output/tmp || exit;
rm -rf output/tmp/*;

# BASE SBATCH FILE
base_sbatch_file="#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --nodelist=${NODE}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
"

# QUEUE
echo;
for graph in $@; do
    if [[ -f ${graph} ]]; then
        matrix_name="$(basename "${graph%.*}")"
        result_folder="output/tmp/${matrix_name}"
        echo "QUEUED JOB for ${matrix_name} (may take some time to be started)"

        sbatch_file="${base_sbatch_file}"
        sbatch_file+=$(printf "\n#SBATCH --output=${result_folder}/output.log")
        sbatch_file+=$(printf "\n#SBATCH --error=${result_folder}/errors.log")
        sbatch_file+=$(printf "\n\n\n./build/main ${graph} ${result_folder}/times.csv")

        echo "${sbatch_file}" | sbatch >/dev/null || exit;
    else
        echo "ERROR file ${graph} doesn't exists"
    fi
done

# WAIT JOB FINISH
echo -e "\nWAITING FOR JOB TO BE QUEUED AND EXECUTED";
until (( $(squeue -u matteo.dinoia -h | wc -l) == 0)); do printf .; sleep 1; done;
