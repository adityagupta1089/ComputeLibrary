for graph in alexnet googlenet mobilenet squeezenet resnet50; do
    ./build/release/examples/graph_temp_scheduler --graph=$graph --profile-temp
done