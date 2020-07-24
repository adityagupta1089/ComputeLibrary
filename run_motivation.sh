for graph in resnet50; do
    ./build/release/examples/graph_temp_scheduler2 --graph=$graph --n=600 --i=10
done
