#!/bin/bash

# Initialize one active-learning step.
# This script prepares stock chunks and writes <STEP>_job_al,
# the master SLURM script that drives AL iterations.

if [ $# -lt 1 ] ; then
    echo "    Usage: $0 step_number stock_file_size nnodes"
    echo "   step_number : number of the active learning step (0 if first step)"
    exit 1
fi

STEP_NB=$1
#launch_jobs=$2

workdir=$(pwd)

### Define necessary directories
NAPSDIR="$(cd "$(dirname "$0")" && pwd)"
ROOTDIR="$(cd "${NAPSDIR}/.." && pwd)"
ALDIR=${ROOTDIR}/ActiveLearning

# put path to env.py
ND=$(echo "$NAPSDIR"|sed 's/\//\\\//g')
cp ${ALDIR}/environment.py ${ALDIR}/env.py
sed -i "s/MYBIN/$ND/" ${ALDIR}/env.py


nNNP=$($NAPSDIR/read_input.sh input_AL NB_NNP)
nsdata=$($NAPSDIR/read_input.sh input_AL STOCK_FILE_SIZE)
nnodes=$($NAPSDIR/read_input.sh input_AL NB_NODES_AL)
nALcycles=$($NAPSDIR/read_input.sh input_AL NB_AL_CYCLES)
nadd=$($NAPSDIR/read_input.sh input_AL NADD)
if [ -z "$nadd" ] ; then
    nadd=30
fi

# Required local files in current workdir:
# - input_AL    : NAPS parameters
# - python.env  : Python environment/module setup
# - job.env     : SBATCH header template (contains NNODES placeholder)
# - <STEP>_STOCK.data and <STEP>_input.data
# - input1.nn, input2.nn, ... inputN.nn

# Check if python environment is defined in python.env file
# Load it if present
if [ ! -f python.env ] ; then
    echo please provide python.env file : load desired python version
    exit
fi
source ./python.env

# Perform initial task of AL procedire : create needed files and directories
echo "creating stockfiles.data"
${NAPSDIR}/split_stock.sh ${STEP_NB}_STOCK.data ${nsdata}
export PYTHONPATH=$NAPSDIR:$PYTHONPATH
python ${ALDIR}/active_training.py --request=init --stepNB=${STEP_NB} --nnodes=${nnodes} --Nadd=${nadd}

# create job file : job_al
job_file=${STEP_NB}_job_al

cp ${workdir}/job.env $job_file
sed -i "s/NNODES/${nnodes}/" $job_file
cat ${workdir}/python.env >> $job_file
echo "export PYTHONPATH=$NAPSDIR:\$PYTHONPATH
"  >> $job_file

#write for loop 
# <STEP>_job_al loops over AL cycles and for each cycle:
# 1) n2p2 scale/train/predict jobs (parallel)
# 2) NNP comparison and structure selection
# 3) VASP jobs for selected structures (parallel)
# 4) dataset/stock update
echo "NODELIST=\$(scontrol show hostname \$SLURM_NODELIST)" >> $job_file
seq1="1,$((nnodes / 2))"
seq2="$((nnodes / 2 + 1)),$nnodes"
echo "NODE1=\$(echo \"\$NODELIST\" | sed -n \"${seq1}p\")" >> $job_file
echo "export NODE1=\$(echo \$NODE1 | sed \"s/\\ /,/g\")" >> $job_file
echo "NODE2=\$(echo \"\$NODELIST\" | sed -n \"${seq2}p\")" >> $job_file
echo "export NODE2=\$(echo \$NODE2 | sed \"s/\\ /,/g\")" >> $job_file

echo "
for i in {1..${nALcycles}}; do" >> $job_file
echo "    python ${ALDIR}/active_training.py --request=ScalTrainPred --stepNB=${STEP_NB} --nnodes=${nnodes}"  >> $job_file
wout=""
for i in $(seq 1 $nNNP) ; do
#    echo "    if [ -f ./job_STP${i} ] ; then
#        ./job_STP${i} &
#        pid${i}=\$!
#    fi" >> $job_file
    echo "    ./${STEP_NB}_job_STP${i} &" >> $job_file
    echo "    pid${i}=\$!" >> $job_file
    wout+="\$pid${i} "
done
echo "    wait ${wout}" >> $job_file
for i in $(seq 1 $nNNP) ; do
    echo "    rm -f ./${STEP_NB}_job_STP${i}" >> $job_file
done
echo "    echo -e \"│  [\$(date +%T)]   └──▶︎ Done\\n│\" >> ${STEP_NB}_status.log
    ds_increment=0
    ds=1
    while [ \"\$ds\" != \"0\" ]; do
        python ${ALDIR}/active_training.py --request=CompEfSSDFT --stepNB=${STEP_NB} --ds_increment=\${ds_increment} --nnodes=${nnodes}" >> $job_file

wout=""
for i in $(seq 1 $nnodes) ; do
#    echo "    if [ -f ./job_vasp${i} ] ; then
#        ./job_vasp${i} &
#        pid${i}=\$!
#    fi" >> $job_file
    echo "        ./${STEP_NB}_job_vasp${i} &" >> $job_file
    echo "        pid${i}=\$!" >> $job_file
    wout+="\$pid${i} "
done
echo "        wait ${wout}" >> $job_file
for i in $(seq 1 $nnodes) ; do
    echo "        rm -f ./${STEP_NB}_job_vasp${i}" >> $job_file
done
echo "        echo -e \"│  [\$(date +%T)]   └──▶︎ Done\\n│\" >> ${STEP_NB}_status.log
        python ${ALDIR}/active_training.py --request=updates --stepNB=${STEP_NB}  --nnodes=${nnodes}
        ds=\$(tail -2 ${STEP_NB}_status.log|grep -c \"Redo a new : Compare NNPs - get structrs from stock - perform DFT calcs\")
        ds_increment=\$((ds_increment + ds))
        echo -e \"│  [\$(date +%T)]   └──▶︎ Done\\n│\" >> ${STEP_NB}_status.log
        echo -e \"├──────────────────────────────────────────────────────────────────────────────────\" >> ${STEP_NB}_status.log
    done
done
" >> $job_file
chmod u+x $job_file

#if [[ "$launch_jobs" == "YES" ]]; then
#    echo "launch job now"
#    /usr/bin/sbatch ${jobfile}
#    echo " job launched !!!"
#fi


