#!/bin/bash 

# Build and optionally submit MD-test jobs from a trained NNP directory.
# For each LAMMPS data file and (T, box variation), this script:
# 1) creates input_lammps_* files,
# 2) runs LAMMPS,
# 3) extracts extrapolative structures into POSCARs.

if [ $# -lt 2 ]; then
        echo
        echo "   WARNING USAGE SHOULD BE: $0 step_number epoch X YES/NO atomsAlements lammps_input lammps_input.dat "
        echo "   1. step_number : number of the active learning step (0 if first step)"
        echo "   2. epoch nb for the NNP parameters: weights*.data"
 #       echo "   3. X : nb of EW structure to be converted to POSCARS for each LAMMPS trajectory"
 #       echo "   4. YES/NO : launch the jobs that have been created ?"
 #       echo "   5. atoms for NNP: list atoms of the system, ex : HOAlSi"
 #       echo "   6. lammps_input : LAMMPS input data file (instructions file)"
 #       echo "   7-... lammps_input.dat : list of LAMMPS input data files (structure files)"
        echo
        exit
fi

NAPSDIR=$(which $0)
NAPSDIR=${NAPSDIR%/*}

### Check Active Learning input file exists 
if [ ! -s input_AL ]; then
    echo "   input_AL file not found, please create it with the necessary parameters"a
    $NAPSDIR/read_input.sh
    exit 1
fi

###  Define necessary variables
WORKDIR=$(pwd)
STEP_NB=$1
epochnb=$2
launch_jobs=$3

NSTRUCT=$($NAPSDIR/read_input.sh input_AL EW2VASP)
NSTEPS_B2EW=$($NAPSDIR/read_input.sh input_AL NSTEPS_B2EW)
atnnp=$($NAPSDIR/read_input.sh input_AL NNP_ELEMENTS)

lammps_inputfile=${WORKDIR}/$($NAPSDIR/read_input.sh input_AL LAMMPS_INPUT)
if [ ! -f $lammps_inputfile ]; then
        echo "   $lammps_inputfile file does not exist, please provide a valid LAMMPS input file (instructions file)"
        exit 1
fi


#files=$($NAPSDIR/read_input.sh input_AL LAMMPS_DAT_FILES)
#read -ra lammps_datfiles <<< "$files"
#for FILE in ${lammps_datfiles[@]}; do
lammps_datfiles=$($NAPSDIR/read_input.sh input_AL LAMMPS_DAT_FILES)
for FILE in ${lammps_datfiles}; do
    if [ ! -f ${WORKDIR}/${FILE} ]; then
        echo "${FILE} file not found, please create it with the necessary environment variables"
        exit 1
    fi    
done

### Check if env files exist for slurm_jobs lammps and python
FILE=$($NAPSDIR/read_input.sh input_AL JOB_ENV)
if [ ! -f ${WORKDIR}/${FILE} ]; then
    echo "${FILE} file not found, please create it with the necessary environment variables"
    exit 1
fi
FILE=$($NAPSDIR/read_input.sh input_AL LAMMPS_ENV)
if [ ! -f ${WORKDIR}/${FILE} ]; then
    echo "${FILE} file not found, please create it with the necessary environment variables"
    exit 1
fi
FILE=$($NAPSDIR/read_input.sh input_AL PYTHON_ENV)
if [ ! -f ${WORKDIR}/${FILE} ]; then
    echo "${FILE} file not found, please create it with the necessary environment variables"
    exit 1
fi
#echo ${lammps_datfiles[@]}
#echo $lammps_inputfile


#exit 1

### Define variations for temperature and box dimensions for the MD Tests
maxEW=$($NAPSDIR/read_input.sh input_AL MAX_EW)
tempvariations=$($NAPSDIR/read_input.sh input_AL TEMP)
xvariations=$($NAPSDIR/read_input.sh input_AL CELL_VAR_X)
yvariations=$($NAPSDIR/read_input.sh input_AL CELL_VAR_Y)
zvariations=$($NAPSDIR/read_input.sh input_AL CELL_VAR_Z)

read -r -a temp_array <<< "$tempvariations"
ntemps=${#temp_array[@]}
if [ "$ntemps" -eq 0 ]; then
    echo "   TEMP is empty in input_AL; please provide at least one temperature"
    exit 1
fi

if [ "$ntemps" -eq 1 ]; then
    T0=${temp_array[0]}
    T1=${temp_array[0]}
    temp_label="${T0}"
    echo "   MD tests at fixed temperature: ${T0} K"
else
    T0=${temp_array[0]}
    T1=${temp_array[0]}
    for t in "${temp_array[@]}"; do
        if [ "$t" -lt "$T0" ]; then
            T0=$t
        fi
        if [ "$t" -gt "$T1" ]; then
            T1=$t
        fi
    done
    temp_label="${T0}to${T1}"
    if [ "$ntemps" -gt 2 ]; then
        echo "   WARNING: TEMP has more than 2 values; using ramp between extremes: ${T0} K -> ${T1} K"
    else
        echo "   MD tests with temperature ramp: ${T0} K -> ${T1} K"
    fi
fi


### Define necessary directories
TRAINDIR=${WORKDIR}/${STEP_NB}_TRAIN
# Check if the training directory exists
if [ ! -d ${TRAINDIR} ]; then
        echo "   ${TRAINDIR} directory does not exist, perform a train before running this script"
        exit 1
fi
        
# For each lammps_datfiles
for lmpdatfile in ${lammps_datfiles}; do
        lammps_datfile_path=${WORKDIR}/${lmpdatfile}
        # create directory MDruns or clean it
        MDTESTDIR=${TRAINDIR}/MDs_${lmpdatfile%.*}
        MDRUNSDIR=${MDTESTDIR}/MDruns
        if [ ! -d ${MDRUNSDIR} ]; then
                mkdir -p ${MDRUNSDIR}
                echo "   ${MDRUNSDIR} directory created"
        else
                rm -rf ${MDRUNSDIR}/*
                echo "   ${MDRUNSDIR} directory already exists, delete all files in it"
        fi

        # Create VASP directory or clean it
        VASPDIR=${MDTESTDIR}/EW_poscar
        if [ ! -d ${VASPDIR} ]; then
                mkdir ${VASPDIR}
                echo "   ${VASPDIR} directory created"
        else
                rm -rf ${VASPDIR}/*
                echo "   ${VASPDIR} directory already exists, delete all files in it"
        fi

        ### Copy necessary files
        cd ${TRAINDIR}
        for file in weights.*.0*${epochnb}.out; do
            wfile=${file%.*.*}
            cp $file ${wfile}.data
        done
        mkdir -p ${MDRUNSDIR}/${temp_label}K
        echo "   ${temp_label}K directory created in ${MDRUNSDIR}"
        cp input.nn scaling.data ${lammps_datfile_path} ${lammps_inputfile} weights*.data ${MDRUNSDIR}/${temp_label}K
        echo "   n2p2 files: input.nn scaling.data weights*.data and ${lammps_datfile_path} ${lammps_inputfile} copied to ${MDRUNSDIR}/${temp_label}K"

        ### Create input_lammps files according to temp,x ,y ,z variations
        cd ${MDRUNSDIR}/${temp_label}K
        for Xvar in ${xvariations}; do
            Xfactor=$(awk -v v="$Xvar" 'BEGIN { printf "%.2f", 1 + v / 100 }')
            for Yvar in ${yvariations}; do
                Yfactor=$(awk -v v="$Yvar" 'BEGIN { printf "%.2f", 1 + v / 100 }')
                for Zvar in ${zvariations}; do
                    Zfactor=$(awk -v v="$Zvar" 'BEGIN { printf "%.2f", 1 + v / 100 }')
                    suffix=${temp_label}_${Xfactor}x${Yfactor}y${Zfactor}z
                    lif=input_lammps_${suffix}
                    cp ${lammps_inputfile} $lif
                    sed -i "s/T0TEMP/${T0}/" $lif
                    sed -i "s/T1TEMP/${T1}/" $lif
                    sed -i "s/MEW/${maxEW}/" $lif
                    sed -i "s/suffix/${suffix}/" $lif
                    sed -i "s/LAMMPSINPUTDAT/${lmpdatfile}/" $lif
                    if [ "$Xfactor" != "1.00" ]; then
                            sed -i "s/#change_box     all x scale xxx remap/change_box     all x scale ${Xfactor} remap/" $lif
                    fi
                    if [ "$Yfactor" != "1.00" ]; then
                            sed -i "s/#change_box     all y scale yyy remap/change_box     all y scale ${Yfactor} remap/" $lif
                    fi
                    if [ "$Zfactor" != "1.00" ]; then
                            sed -i "s/#change_box     all z scale zzz remap/change_box     all z scale ${Zfactor} remap/" $lif
                    fi
                done
            done
        done
        echo "   input_lammps files created in ${DIR} with variations for temperature and box dimensions"


        ### Create jobs to perform the lammps MDs on the input_lammps files 
        cd ${WORKDIR}

        for Xvar in ${xvariations}; do
            Xfactor=$(awk -v v="$Xvar" 'BEGIN { printf "%.2f", 1 + v / 100 }')
            jobfile=${MDRUNSDIR}/${temp_label}K/job_${temp_label}_${Xfactor}x
            cat job.env lammps.env > ${jobfile}
            sed -i "s/NNODES/1/g" ${jobfile}
            sed -i "s/jobname/MD_${temp_label}_${Xfactor}x/g" ${jobfile}
            echo "cd ${MDRUNSDIR}/${temp_label}K
for inplammps in input_lammps_${temp_label}_${Xfactor}x*; do
    srun lmp < \${inplammps} > \${inplammps}.out
done

# extract EW structures to create POSCARS" >> ${jobfile}
            cat python.env >> ${jobfile}
            echo "export PYTHONPATH=${NAPSDIR}:\$PYTHONPATH
for inplammps in input_lammps_${temp_label}_${Xfactor}x*.out; do
    name=\${inplammps%.out}
    name=\${name#input_lammps_}
    python ${NAPSDIR}/xget-EW-structures1.py --dumpfile=dump_\${name}.lammpstrj --outlammpsfile=input_lammps_\${name}.out --atomsNNP=${atnnp} --prefix=\${name} --nbstruct=$NSTRUCT --step=$NSTEPS_B2EW
done
" >> ${jobfile}
        done


        echo "   LAMMPS jobs created in directory: MDruns"

        ### Launch jobs if requested 
        if [ "$launch_jobs" == "YES" ]; then
            cd ${MDRUNSDIR}/${temp_label}K
            for boj in job_*00_*x; do
                    sbatch $boj
            done
            cd ..
        fi
done