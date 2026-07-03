#!/bin/bash 

# Read one parameter value from input_AL.
# Expected line format in input_AL:
# PARAM_NAME value1 value2 ...

if [ $# -lt 2 ]; then
        echo
        echo "   WARNING USAGE SHOULD BE: $0 Active_input_file parameter "
        echo "   Active_input_file : file containing all parameters"
        echo "   Parameter : the expected parameter read in the Active_input_file"
        echo "   Parameters may be:"
        echo "Defining Environments  :        * PYTHON_ENV        * JOB_ENV          * LAMMPS_ENV"
        echo "                       :        * VASP_ENV          * N2P2_ENV"
        echo
        echo "Active Learning      :        * NB_NODES_AL       * STOCK_FILE_SIZE   * NB_NNP"
        echo "Parameters             :        * NNP_ELEMENTS      * NNP_FILES_AL      * NB_AL_CYCLES"
        echo "                       :        * NADD"
        echo
        echo "Training parameters    :        * NNP_FILE_TRAIN    * NB_NODES_TRAIN"
        echo
        echo "For MD Tests           :        * TEMP              * MAX_EW"
        echo "                                * CELL_VAR_X        * CELL_VAR_Y       * CELL_VAR_Z"
        echo
        echo "LAMMPS FILES           :        * LAMMPS_INPUT      * LAMMPS_DAT_FILES" 
        echo
        echo "MDs EW Convrtd to VASP :        * EW2VASP           * NSTEPS_B2EW"
        echo
        echo "VASP Calc. on EW sruct.:        * NB_NODES_DFT"
        echo
        echo "Create Random Sructures:        * NSTRUCT_DUMP      * NSTRUCT_RAND"
        echo "      from dumpfiles"
        echo
        exit
fi

AL_inputfile=$1
param=$2

###  Define necessary variables and check files exist
WORKDIR=$(pwd)


# Declare an associative array to store variables and their values
declare -A values_array

# Read the file line by line
while IFS= read -r line; do
    # Skip lines starting with "#" (comments)
    [[ "$line" =~ ^#.*$ ]] && continue

    # Match lines that start with the requested parameter name.
    if [[ "$line" =~ ^($param) ]]; then
        # Split the line into the variable name and the rest of the values
        read var_name values <<< "$line"
    fi
done < ${AL_inputfile}

echo ${values}
