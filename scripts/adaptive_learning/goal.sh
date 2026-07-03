#!/bin/bash 

# Front-end command for the NAPS workflow.
# Use this script to call the different stages of the AL/MD/DFT pipeline.

if [ $# -lt 1 ]; then
        echo
        echo "   Launch NAPS directives !   Here are the possibilities : "
        echo
        echo "   goal init AL_step_nb"
        echo "   ---- ---- ----------"
        echo "      > to initialiaze parameters, directories ... "
        echo "      >> AL_step_nb : Adaptive Learning step number"
#        echo "      >> launch Adaptive Learning Phase ? YES/NO"
        echo
        echo "   goal sel_inp input.data struct_nb output.data"
        echo "   ---- ------- ---------- --------- -----------"
        echo "      > select the first struct_nb structures from input.data, write them to output.data"
        echo "      >> input.data and output.data have the format of n2p2 data files"
        echo "      >> struct_nb : all structures up to struct_nb will be included in output.data"
        echo
        echo "   goal train AL_step_nb input.data launch"
        echo "   ---- ----- ---------- ---------- ------"
        echo "      > create X_TRAIN directory, copy files for training"
        echo "      >> AL_step_nb : Adaptive Learning step number"
        echo "      >> input.data : n2p2 file containing structures for training"
        echo "      >> launch training ? YES/NO"
        echo
        echo "   goal xMDs AL_step_nb epoch_nb launch"
        echo "   ---- ---- ---------- ---------------"
        echo "      > create MD directories, copy necessary files, create jobs, ..."
        echo "      >> AL_step_nb : Adaptive Learning step number"
        echo "      >> epcoh_nb : weights parameters from training epoch_nb will be taken for n2p2 potential"
        echo "      >> launch MD tests ? YES/NO"
        echo
        echo "   goal get_EWnb AL_step_nb"
        echo "   ---- -------- ----------"
        echo "      > create a file : \${STEP}_EW.txt with the nb of EW per MD test simulation"
        echo "      >> AL_step_nb : Adaptive Learning step number"
        echo
        echo "   goal EW_DFT AL_step_nb launch"
        echo "   ---- ------ ---------- ------"
        echo "      > perform DFT calculations on selected EW structures from MD tests"
        echo "      > creates a new \${STEP}input.data file for a new Adaptive Learning phase"
        echo "      >> AL_step_nb : Adaptive Learning step number"
        echo "      >> launch DFT calculations ? YES/NO"
        echo
        echo "   goal rnw_stck AL_step_nb launch"
        echo "   ---- -------- ---------- ------"
        echo "      > generate random structures from EW structures from MD tests"
        echo "      >> AL_step_nb : Adaptive Learning step number"
        echo "      >> launch jobs ? YES/NO"
        echo
        echo "  goal crt_stck AL_step_nb"
        echo "   ---- ------- ----------"
        echo "      > creates a new \${STEP+1}_STOCK.data file for a new Adaptive Learning phase"
        echo "      >> AL_step_nb : Previous Adaptive Learning step number, this will be implemented for the new adapative phase"
        echo
        echo "  goal div_data inpfile.data div outfile.data"
        echo "   ---- ------- ----------"
        echo "      > divides the number of structures of inpfile.data by div and write to outfile.data"
        echo
        echo "   goal rnw_dataset AL_step_nb old_input.data"
        echo "   ---- ----------- ---------- --------------"
        echo "      > generate the new dataset from old one enriched with DFT computed EW structures"
        echo "      > creates a new \${STEP}_input.data file for the new Adaptive Learning phase"
        echo "      >> AL_step_nb : Current Adaptive Learning step number"
        echo "      >> old_input.data : Old input.data n2p2 file"
        echo
        exit
fi

### Define necessary directories
NAPSDIR=$(which $0)
NAPSDIR=${NAPSDIR%/*}

# Dispatch to stage-specific scripts.
case "$1" in
    "init")
        if [ $# -lt 2 ] ; then
            echo "Missing Arguments"
            exit
        fi
        ${NAPSDIR}/init_adaptive.sh $2
        ;;
    "sel_inp")
        if [ $# -lt 4 ] ; then
            echo "Missing Arguments"
            exit
        fi
        ${NAPSDIR}/cut_inpdat.sh $2 $3 $4
        ;;
    "train")
        if [ $# -lt 4 ] ; then
            echo "Missing Arguments"
            exit
        fi
        ${NAPSDIR}/xtrain $2 $3 $4
        ;;
    "xMDs")
        if [ $# -lt 4 ] ; then
            echo "Missing Arguments"
            exit
        fi
        ${NAPSDIR}/doMDtests.sh $2 $3 $4
        ;;
    "get_EWnb")
        ${NAPSDIR}/get_EWnb.sh $2
        ;;
    "EW_DFT")
        if [ $# -lt 3 ] ; then
            echo "Missing Arguments"
            exit
        fi
        ${NAPSDIR}/dovasp.sh $2 $3
        ;;
    "rnw_stck")
        if [ $# -lt 3 ] ; then
            echo "Missing Arguments"
            exit
        fi
        ${NAPSDIR}/renewstock.sh $2 $3
        ;;
    "crt_stck")
        if [ $# -lt 2 ] ; then
            echo "Missing Arguments"
            exit
        fi
        ${NAPSDIR}/createstock.sh $2
        ;;
    "div_data")
        if [ $# -lt 4 ] ; then
            echo "Missing Arguments"
            exit
        fi
        ${NAPSDIR}/divdata.sh $2 $3 $4
        ;;
    "rnw_dataset")
        if [ $# -lt 3 ] ; then
            echo "Missing Arguments"
            exit
        fi
        ${NAPSDIR}/renewdataset.sh $2 $3
        ;;
    *)
        # if argument 1 is not in the list
        echo "$1 not in the authorized list"
        exit 1
        ;;
esac


