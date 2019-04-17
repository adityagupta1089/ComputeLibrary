#!/bin/bash
export LD_LIBRARY_PATH=./build/release/
ns=(10 10 100)
is=(10 100 1)
length=${#ns[@]}
for graph in "graph_alexnet_2" "graph_mobilenet_2" "graph_googlenet_2" "graph_squeezenet_2" "graph_resnet50_2"; do
    for target in "--cpu" "--gpu" "--cpu --gpu"; do
        for ((x=0;x<$length;x++)); do
            cmd="./build/release/examples/$graph $target --n=${ns[$x]} --i=${is[$x]}"
            echo $cmd
            eval $cmd
        done
    done
done
