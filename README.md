This is a fork of the original library to be able to be run on **Hikey 970** board using a `makefile` on top of `scons`.

# Instructions
- Run `make release` for compiling a release build. `make debug` makes a debug build and `make all` builds both.
- After that `export LD_LIBRARY_PATH=./build/release` to set library path for release build, similarly for debug build.
- For running specific examples `./build/release/graph_alexnet <parameters>`

# Utilizing multiple cores
| Target | Command | 
| -- | -- |
| Small Cores | `taskset -c 0-3 ./graph_alexnet --threads=4 --target=NEON` |
| Big Cores | `taskset -c 4-7 ./graph_alexnet --threads=4 --target=NEON` |
| All Cores | `taskset -c 0-7 ./graph_alexnet --threads=8 --target=NEON` |
| GPU | `taskset -c 0-3 ./graph_alexnet--target=CL` |

# CPU+GPU Utilization
There was a python wrapper ([the last version of file](https://github.com/adityagupta1089/ComputeLibrary/blob/f307d9f03922a81ad525bb2d89aba8fec136c3b2/run.py) before being removed) for utilizing both CPU and GPU but it has been superseeded by modifications to individual graphs named `graph_alexnet_2`, `graph_googlenet_2`, `graph_mobilenet_2`, `graph_resnet50_2`, `graph_squeezenet_2`. They can be run using:
```./graph_alexnet_2 [--cpu] [--gpu] [--n=N] [--i=I]```
where `--cpu` and `-gpu` selects target combinations and `n` are the total images, `i` the total inferences per image.

# Performance Modeling
This is done via `perf.py` and results in `perf_results` folder. Results can be found in `perf_cache.dat`. It runs performance modeling for [graphs](https://github.com/adityagupta1089/ComputeLibrary/blob/d4eabb71064b91f534a4bc2d0086f2dbb883e909/perf.py#L18) and [targets](https://github.com/adityagupta1089/ComputeLibrary/blob/d4eabb71064b91f534a4bc2d0086f2dbb883e909/perf.py#L25) with `n` and `i` values [looped](https://github.com/adityagupta1089/ComputeLibrary/blob/d4eabb71064b91f534a4bc2d0086f2dbb883e909/perf.py#L96). It uses the curve fitting [function](https://github.com/adityagupta1089/ComputeLibrary/blob/d4eabb71064b91f534a4bc2d0086f2dbb883e909/perf.py#L107) and caches in `perf_cache.dat` and finally plotting in `perf_results`.

# Temperature Modeling
This is done via `temp.py` and results in `temp_results` folder. Results can be found in `temp_results.log`. Similar to performance modeling it loops over graphs and targets and uses curve fitting [function](https://github.com/adityagupta1089/ComputeLibrary/blob/d4eabb71064b91f534a4bc2d0086f2dbb883e909/temp.py#L34) to plot a fit along with the threshold temperature 65000. Finally resultant plots are plotted in `temp_plots`. `temp_results.log` was manually created by piping the results of `temp.py`
