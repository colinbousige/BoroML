#!/bin/bash

bwnb=$(sed '/#/d' learning-curve.out | awk '{printf("%0.12f %d \n",$2,$1);}' | sort -n -k1 | head -1 | awk '{printf("%d",$2);}')
npred=$(ls -l ..|grep -c predict )

for file in weights.*${bwnb}.out ; do
    wfile=${file%.*}
    wfiledata=${wfile%.*}.data
    for i in $(seq $npred) ; do 
        cp ${file} ../predict${i}/${wfiledata}
    done
done

for i in $(seq $npred) ; do
    cp input.nn ../predict${i}
    cp scaling.data ../predict${i}
    sed -i 's/epochs/epochs 0 \#/' ../predict${i}/input.nn
    sed -i 's/test_fraction/test_fraction 0 \#/' ../predict${i}/input.nn
    sed -i 's/\#use_old_weights_short/use_old_weights_short/' ../predict${i}/input.nn
    sed -i 's/normalize_data_set/\#normalize_data_set/' ../predict${i}/input.nn
    sed -i 's/write_trainpoints/write_trainpoints 1 \#/' ../predict${i}/input.nn
    sed -i 's/write_trainforces/write_trainforces 1 \#/' ../predict${i}/input.nn
done
