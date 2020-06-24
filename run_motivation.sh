for graph in googlenet mobilenet squeezenet alexnet resnet50; do
    ./build/release/examples/graph_temp_scheduler2 --graph=$graph --n=200
done
