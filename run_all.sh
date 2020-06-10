for graph in alexnet googlenet mobilenet resnet50 squeezenet; do
    for version in 3 3.1 4; do
        for tl in 80000 999999; do
            ./build/release/examples/graph_temp_scheduler --graph=$graph --run-sched --version=$version --tl=$tl
        done
    done
done
