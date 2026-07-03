#!/bin/bash

if [ $# -lt 2 ] ; then
    echo "Usage: $0 step number input.data "
    echo "       step number"
    echo "       input.data file containing to be increases with vasp computed EW structures"
    exit 1
fi


###  Define necessary variables
WORKDIR=$(pwd)
STEP_NB=$1
NEW_STEP_NB=$(($1 + 1))
inpdatfile=$2
newinpdatfile=${NEW_STEP_NB}_input.data

NAPSDIR=$(which $0)
NAPSDIR=${NAPSDIR%/*}
lammps_datfiles=$($NAPSDIR/read_input.sh input_AL LAMMPS_DAT_FILES)

### Check inpdatfile file exists
if [ ! -f ${inpdatfile} ]; then
    echo "   ${inpdatfile} file does not exist, can't proceed without it"
    exit 1
fi

# check if newinpdatfile exists and delete it if YES
if [ ! -f $newinpdatfile ]; then
    echo "   Old ${newinpdatfile} file has been deleted"
fi

# create new inpdatfile
cp $inpdatfile $newinpdatfile
nstruct=$(grep -c begin ${newinpdatfile})
echo "   New ${newinpdatfile} has been created, it contains ${nstruct} structures"

### Check env file exists
if [ ! -f python.env ]; then
        echo "   python.env file does not exist, please provide it !"
        exit 1
fi

### Check necessary directory exist
TRAINDIR=${WORKDIR}/${STEP_NB}_TRAIN
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
        VASPDIR=${MDTESTDIR}/EW_poscar
        if [ ! -d ${VASPDIR} ]; then
                echo "   ${VASPDIR} directory does not exist, perform a train, MD tests and Vasp calcs before running this script"
                exit 1
        fi
done

source python.env
export PYTHONPATH=${NAPSDIR}:$PYTHONPATH

for lmpdatfile in ${lammps_datfiles}; do
        echo "    For Lammps dat file : ${lmpdatfile}"
        MDTESTDIR=${TRAINDIR}/MDs_${lmpdatfile%.*}
        VASPDIR=${MDTESTDIR}/EW_poscar
        cd ${VASPDIR}
        # clean directory
        if [ -f convert.out ]; then 
                rm -f OUTCAR*data convert.out
        fi
        # create .datafiles from OUTCARs
        echo "    Convert OUTCARS to n2p2 inpdat files"
        for FILE in OUTCAR*; do
        ext=${FILE#*_}
        python ${NAPSDIR}/convert-VASP_OUTCAR_AU.py $FILE OUTCAR_${ext}.data 2>> convert.out
        done
        NOUT=$(ls -l OUTCAR_*data | wc -l)
        echo "   ${NOUT} OUTCARs converted to n2p2 data files"
        # Delete empty OUTCAR.inp because forces > 25 eV/A
        find ./ -maxdepth 1 -name "OUTCAR*" -type f -size 0 -delete
        NOUT=$(ls -l OUTCAR_*data | wc -l)
        echo "   ${NOUT} OUTCARs converted to n2p2 data files after deleted those with forces > 25 eV/A"

        # put all structures in :
        cat *.data > OUTCARs.data
        rm -f OUTCAR_*data

        cd ${WORKDIR}
        cat ${VASPDIR}/OUTCARs.data >> ${newinpdatfile}
        nstruct=$(grep -c begin ${newinpdatfile})

        echo "   ${nstruct} structures in ${newinpdatfile}"
done