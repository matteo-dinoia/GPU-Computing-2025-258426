#!/usr/bin/bash
# Always run this script from the root of the deliverable2

echo "COMPILING";
make normal || exit;

# PRE RUN
mkdir -p output/tmp || exit;
rm -rf output/tmp/*;

# RUN
for graph in $@; do
    if [[ -f ${graph} ]]; then
        matrix_name="$(basename "${graph%.*}")"
        result_folder="output/tmp/${matrix_name}"

        echo -e "\nRUNNING LOCALLY ${matrix_name}, may take very long to finish (depending on matrix size)";
        mkdir -p ${result_folder}
        ./build/main ${graph} ${result_folder}/times.csv > >(tee "${result_folder}/ouput.log") 2> >(tee "${result_folder}/errors.log" >&2)
    else
        echo "ERROR file ${graph} doesn't exists"
    fi

    echo -e "\n\n"
done;

