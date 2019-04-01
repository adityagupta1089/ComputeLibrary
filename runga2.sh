#!/bin/bash
ns=(10 10 100)
is=(10 100 1)
length=${#ns[@]}
for target in "--cpu" "--gpu" "--cpu --gpu"; do
    for ((x=0;x<$length;x++)); do
        cmd="./build/release/examples/graph_alexnet_2 $target --n=${ns[$x]} --i=${is[$x]}"
        echo $cmd
        eval $cmd
    done
done
