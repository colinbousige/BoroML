#!/bin/bash

if [ $# -lt 1 ] ; then
    echo "Usage: $0 step_number"
    exit 1
fi


###  Define necessary variables
WORKDIR=$(pwd)
STEP_NB=$1
NEW_STEP_NB=$(($STEP_NB + 1))
NEW_STOCK_FILE=${WORKDIR}/${NEW_STEP_NB}_STOCK.data
if [ -f ${NEW_STOCK_FILE} ]; then
    echo
    echo "   WARNING: ${NEW_STOCK_FILE} file already exists, delete it before running this script"
    echo
    exit 1
fi

### Check necessary directory exist
NAPSDIR="$(cd "$(dirname "$0")" && pwd)"
lammps_datfiles=$($NAPSDIR/read_input.sh input_AL LAMMPS_DAT_FILES)
tempvariations=$($NAPSDIR/read_input.sh input_AL TEMP)

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


#   Take strcuture from dump file every 10 structures
for lmpdatfile in ${lammps_datfiles}; do
    echo "    For Lammps dat file : ${lmpdatfile}"
    MDTESTDIR=${TRAINDIR}/MDs_${lmpdatfile%.*}
    MDRUNSDIR=${MDTESTDIR}/MDruns
    for temp in ${tempvariations}; do
        echo "    Gathering structures for temperature ${temp}K"
        cat ${MDRUNSDIR}/${temp}K/dumps_*.data >> ${NEW_STOCK_FILE}
        cat ${MDRUNSDIR}/${temp}K/rands_*.data >> ${NEW_STOCK_FILE}

    done
done
nstruts=$(grep -c begin ${NEW_STOCK_FILE})
echo
echo "   ${NEW_STOCK_FILE} file created with structures from dumps and random files"
echo "   There are ${nstruts} structures in this file"
echo


