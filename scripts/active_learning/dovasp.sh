#!/bin/bash 

# Prepare and optionally submit VASP jobs on POSCARs extracted from MD tests.
# OUTCAR_<id> files are generated in EW_poscar/ and later converted to n2p2 data.

if [ $# -lt 2 ]; then
        echo
        echo "   WARNING USAGE SHOULD BE: $0 step_number YES/NO "
        echo "   step_number : number of the active learning step (0 if first step)"
        echo "   YES/NO : launch the jobs that have been created ?"
        echo
        exit
fi

###  Define necessary variables and check files exist
WORKDIR=$(pwd)
if [ ! -f INCAR ] || [ ! -f POTCAR ] || [ ! -f KPOINTS ]; then
        echo "   VASP files INCAR POTCAR KPOINTS do not exist, please provide them !"
        exit 1
fi
if [ ! -f job.env ] || [ ! -f vasp.env ]; then
        echo "   job.env or vasp.env files do not exist, please provide them !"
        exit 1
fi
STEP_NB=$1
launch_jobs=$2

### Define necessary directories
NAPSDIR=$(which $0)
NAPSDIR=${NAPSDIR%/*}

nbnodes=$($NAPSDIR/read_input.sh input_AL NB_NODES_DFT)
lammps_datfiles=$($NAPSDIR/read_input.sh input_AL LAMMPS_DAT_FILES)

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
        # Check if directories MDs do exist
        if [ ! -d ${MDTESTDIR} ]; then
                echo "   ${MDTESTDIR} directory does not exist, perform a train before running this script"
                exit 1
        fi
        # Create VASP directory or clean it
        VASPDIR=${MDTESTDIR}/EW_poscar
        if [ ! -d ${VASPDIR} ]; then
                mkdir ${VASPDIR}
                echo "   ${VASPDIR} directory created"
        fi

        ### Create job and input files for VASP calculations
        cd $WORKDIR
        cp INCAR POTCAR KPOINTS ${VASPDIR}
        jobfile=${VASPDIR}/job_VASP
        cat job.env vasp.env> ${jobfile}
        sed -i "s/NNODES/$nbnodes/g" ${jobfile}
        sed -i "s/jobname/vasp/g" ${jobfile}
        echo "cd ${VASPDIR}

        for FILE in POSCAR*
        do
        ext=\${FILE#*_}
        cp \$FILE POSCAR
        srun vasp_std
        mv OUTCAR OUTCAR_\${ext}
        rm -f PCDAT CHG XDATCAR out VASP.* ICONST IBZKPT REPORT EIGENVAL DOSCAR vaspout.h5 CHGCAR WAVECAR OSZICAR* CONTCAR* vasprun*.xml
        done
        " >> ${jobfile}
        chmod u+x ${jobfile}


        echo "   VASP job created in directory: ${VASPDIR}"

        if [ "$launch_jobs" == "YES" ]; then
        cd ${VASPDIR}
        sbatch ${jobfile}
        fi
done