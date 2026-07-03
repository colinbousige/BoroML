#!/bin/bash

STEP=$1
EWfile=${STEP}_EW.txt
nEW=0

rm -f $EWfile
touch $EWfile

for mat in Gib Kao PYRO
do
        DIR=${STEP}_TRAIN/MDs_${mat}/MDruns
        for file in ${DIR}/*00K/input*out
        do
                new=$(grep "EXTRAPOLATION WARNING"  $file|wc -l)
                line=("$file $new")
                fin=${line: -2}
                if [ $fin -ne " 0" ]
                then
                        echo $line
                        echo $line >> $EWfile
                        newi=$(echo $line|awk '{ print $2}')
                        nEW=$(( $nEW + $newi ))
                fi
        done
done
echo "Total nb of of EW: ${nEW}"
echo "Total nb of of EW: ${nEW}" >> $EWfile
