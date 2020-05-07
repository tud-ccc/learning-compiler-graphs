#!/bin/bash
ALGORITHMS="deeptune inst2vec gnn_ast gnn_llvm grewe random static"
EXPERIMENTS="grouped random"

for EXPERIMENT in $EXPERIMENTS; do
    HEADER=0
    OUTPUT="devmap_$EXPERIMENT.csv"
    for ALGO in $ALGORITHMS; do
        FOLDER="results/devmap/devmap_${ALGO}_${EXPERIMENT}"
        FILES=`ls $FOLDER/*raw.txt`;
        for file in $FILES; do
            if [ $HEADER -eq 0  ]
            then
               echo -n "filename," > $OUTPUT
               head -n 1 $file | cut -d ',' -f 2,3,4,5,6,7,8,9,10,11 >> $OUTPUT
               HEADER=1
            fi;
            filename=${file##*/}
            sed -nE "2,$ s/^/$filename,/p" $file | cut -d ',' -f 1,3,4,5,6,7,8,9,10,11,12 | sed 's/ncc_devmap/inst2vec/g' >> $OUTPUT
        done;
    done;
    sed -i "" 's/DeepGNN/GNN/g' $OUTPUT
done;
