#!/bin/bash 

if [ $# -lt 3 ]; then
        echo
        echo "   WARNING USAGE SHOULD BE: $0 step_number X YES/NO "
        echo "   step_number : number of the active learning step (0 if first step)"
        echo "   X : nb of EW structure to be converted to POSCARS for each LAMMPS trajectory"
        echo "   YES/NO : launch the jobs that have been created ?"
        echo "   atoms for NNP: list atoms of the system, ex : HOAlSi"
        echo
        exit
fi

###  Define necessary variables
WORKDIR=$(pwd)
STEP_NB=$1
NSTRUCT=$2
launch_jobs=$3
atnnp=$4


### Define necessary directories
NAPSDIR=$(which $0)
NAPSDIR=${NAPSDIR%/*}
TRAINDIR=${WORKDIR}/$1_TRAIN
# Check if the training directory exists
if [ ! -d ${TRAINDIR} ]; then
        echo "   ${TRAINDIR} directory does not exist, perform a train before running this script"
        exit 1
fi
# create directory MDruns or clean it
MDTESTDIR=${TRAINDIR}/MDs
MDRUNSDIR=${MDTESTDIR}/MDruns
if [ ! -d ${MDRUNSDIR} ]; then
        echo "   ${MDRUNSDIR} directory does not exists"
fi
# Create VASP directory or clean it
VASPDIR=${MDTESTDIR}/EW_poscar
if [ ! -d ${VASPDIR} ]; then
        mkdir ${VASPDIR}
        echo "   ${VASPDIR} directory created"
fi



### Define variations for temperature and box dimensions for the MD Tests
tempvariations=(300 500 1000)
# xvariations=(0 -5 5 -10 10)
# yvariations=(0 -5 5 -10 10)
# zvariations=(0 -5 5 -10 10)


### Check if necessary files exist
cd ${WORKDIR}
if [ ! -f job.env ]; then
    echo "job.env file not found, please create it with the necessary environment variables"
    exit 1
fi
# if [ ! -f lammps.env ]; then
#     echo "lammps.env file not found, please create it with the necessary N2P2 environment variables"
#     exit 1
# fi
if [ ! -f python.env ]; then
    echo "python.env file not found, please create it with the necessary environment variables"
    exit 1
fi


### Create jobs to perform the extraction of strcutures from LAMMPS trajectories and convert them to POSCARS 
for temp in "${tempvariations[@]}"; do
    jobfile=${MDRUNSDIR}/${temp}K/job_${temp}
    cat job.env python.env > ${jobfile}
    sed -i "s/NNODES/1/g" ${jobfile}
    sed -i "s/jobname/exEW${temp}/g" ${jobfile}
    echo "export PYTHONPATH=${NAPSDIR}:\$PYTHONPATH
        
cd ${MDRUNSDIR}/${temp}K

# extract EW structures to create POSCARS
for inplammps in input_lammps_${temp}*.out; do
    name=\${inplammps%.out}
    name=\${name#input_lammps_}
    python ${NAPSDIR}/xget-EW-structures1.py --dumpfile=dump_\${name}.lammpstrj --outlammpsfile=input_lammps_\${name}.out --atomsNNP=${atnnp} --prefix=\${name} --nbstruct=$NSTRUCT --step=25
done
" >> ${jobfile}
    chmod u+x ${jobfile}
    ### Launch jobs if requested 
    if [ "$launch_jobs" = "YES" ]; then
        cd ${MDRUNSDIR}/${temp}K
        sbatch ${jobfile}
    fi
done



# ### Create jobs to perform the extraction of strcutures from LAMMPS trajectories and convert them to POSCARS 
# for temp in "${tempvariations[@]}"; do
#     for Xvar in "${xvariations[@]}"; do
#         Xfactor=$(awk -v v="$Xvar" 'BEGIN { printf "%.2f", 1 + v / 100 }')
#         jobfile=${MDRUNSDIR}/${temp}K/job_${temp}_${Xfactor}x
#         cat job.env python.env > ${jobfile}
#         sed -i "s/NNODES/1/g" ${jobfile}
#         sed -i "s/jobname/MD_${temp}_${Xfactor}x/g" ${jobfile}
#         echo "export PYTHONPATH=${NAPSDIR}:\$PYTHONPATH
        
# cd ${MDRUNSDIR}/${temp}K

# # extract EW structures to create POSCARS
# for inplammps in input_lammps_${temp}_${Xfactor}x*.out; do
#     name=\${inplammps%.out}
#     name=\${name#input_lammps_}
#     python ${NAPSDIR}/xget-EW-structures1.py --dumpfile=dump_\${name}.lammpstrj --outlammpsfile=input_lammps_\${name}.out --atomsNNP=HOAl --prefix=\${name} --nbstruct=$NSTRUCT --step=25
# done
# " >> ${jobfile}
#         ### Launch jobs if requested 
#         if [ "$launch_jobs" = "YES" ]; then
#             sbatch ${jobfile}
#         fi
#     done
# done

