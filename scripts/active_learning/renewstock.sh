#!/bin/bash

if [ $# -lt 2 ] ; then
    echo "Usage: $0 step_number lammps_input.dat YES/NO "
    echo "       step number"
    echo "       YES/NO : launch the jobs that have been created ?"

    exit 1
fi


###  Define necessary variables
WORKDIR=$(pwd)
STEP_NB=$1
launch_jobs=$2

NEW_STEP_NB=$(($STEP_NB + 1))

NAPSDIR=$(which $0)
NAPSDIR=${NAPSDIR%/*}

NSTRUCT_FROM_DUMP=$($NAPSDIR/read_input.sh input_AL NSTRUCT_DUMP)
NSTRUCT_FROM_RANDOM=$($NAPSDIR/read_input.sh input_AL NSTRUCT_RAND)
lammps_datfiles=$($NAPSDIR/read_input.sh input_AL LAMMPS_DAT_FILES)
atnnp=$($NAPSDIR/read_input.sh input_AL NNP_ELEMENTS)

#   for the loop over the x variations : to allow less long jobs
xvariations=$($NAPSDIR/read_input.sh input_AL CELL_VAR_X)


### Check env files exist
if [ ! -f job.env ] || [ ! -f python.env ]; then
        echo "   job.env or python.env files do not exist, please provide them !"
        exit 1
fi

### Check necessary directory exist
TRAINDIR=${WORKDIR}/$1_TRAIN
# Check if the training directory exists
if [ ! -d ${TRAINDIR} ]; then
        echo "   ${TRAINDIR} directory does not exist, perform a train before running this script"
        exit 1
fi
for lmpdatfile in ${lammps_datfiles}; do
    MDTESTDIR=${TRAINDIR}/MDs_${lmpdatfile%.*}
    if [ ! -d ${MDTESTDIR} ]; then
            echo "   ${MDTESTDIR} directory does not exist, perform a trainand MD tests before running this script"
            exit 1
    fi
    MDRUNSDIR=${MDTESTDIR}/MDruns
    if [ ! -d ${MDRUNSDIR} ]; then
            echo "   ${MDRUNSDIR} directory does not exist, perform a trainand MD tests runs before running this script"
            exit 1
    fi
done


for lmpdatfile in ${lammps_datfiles}; do
    # create directory MDruns or clean it
    MDTESTDIR=${TRAINDIR}/MDs_${lmpdatfile%.*}
    MDRUNSDIR=${MDTESTDIR}/MDruns
    cd ${MDRUNSDIR}
    #   Take strcuture from dump file every 10 structures
    for tempdir in $(ls -d *00K); do
        for Xvar in ${xvariations}; do
            Xfactor=$(awk -v v="$Xvar" 'BEGIN { printf "%.2f", 1 + v / 100 }')
            cd ${tempdir}
            cp ${WORKDIR}/${lmpdatfile} .
        #    echo " \$(date +%T)   Build dump.data file from \${tempdir}dump_*.lammpstrj"
            jobfile=${MDRUNSDIR}/${tempdir}/job_mkstock_${Xfactor}x
            cat ${WORKDIR}/job.env ${WORKDIR}/python.env > ${jobfile}
            sed -i "s/NNODES/1/g" ${jobfile}
            sed -i "s/jobname/ms${tempdir}${Xfactor}x/g" ${jobfile}
            echo "export PYTHONPATH=${NAPSDIR}:\$PYTHONPATH

cd ${MDRUNSDIR}/${tempdir}
rm -f Rand*data rands_*.data Dump_*.data dumps_*.data

for file in dump*${Xfactor}x*.lammpstrj; do
    prefix=\${file%.lammpstrj}
    prefix=\${prefix#dump_}
    # Take the last ${NSTRUCT_FROM_DUMP} strcutures from dump file every freq structures
    echo \" \$(date +%T)   Build dump.data file from \${file}\"
    python ${NAPSDIR}/dump2data.py --dumpfile=\${file} --atomsNNP=${atnnp} --prefix=\${prefix} --freq=10 --NGetstruct=${NSTRUCT_FROM_DUMP} --chkfile=${lmpdatfile}
    # Create 10 NRandstruct random structures for each ${NSTRUCT_FROM_RANDOM} NGetstruct last structures with frequency of freq 5
    echo \" \$(date +%T)   Build Random structures from  \${file}\"
    python ${NAPSDIR}/make-rand-st.py --dumpfile=\${file} --NGetstruct=${NSTRUCT_FROM_RANDOM} --freq=5 --maxNorm=0.1 --NRandstruct=10 --atomsNNP=${atnnp} --outfile=Rand_\${prefix}.data --chkfile=${lmpdatfile}
done

cat Dump*${Xfactor}x*.data > dumps_${Xfactor}x.data
rm -f Dump*${Xfactor}x*.data
cat Rand*${Xfactor}x*.data > rands_${Xfactor}x.data
rm -f Rand*${Xfactor}x*.data
" >> ${jobfile}
            chmod u+x ${jobfile}
            echo "   job created in directory: ${MDRUNSDIR}/${tempdir}"
            cd ..
            if [ "$launch_jobs" == "YES" ]; then
                sbatch ${jobfile}
            fi
        done
    done
done
    ### Launch jobs if requested
#    if [ "$launch_jobs" = "YES" ]; then
#        cd ${MDRUNSDIR}
#        for tempdir in $(ls -d *00K); do
#                cd ${tempdir}
#                for job in job_mkstock_*x ; do
#                    sbatch $job
#                done
#                cd ..
#        done
#    fi
#done
