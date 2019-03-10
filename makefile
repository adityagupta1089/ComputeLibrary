all:
	scons -j8 -Q arch=arm64-v8a build=native neon=True opencl=True
al2:
	g++ -o build/examples/graph_alexnet_2.o -c -Wno-deprecated-declarations -Wall -DARCH_ARM -Wextra -Wno-unused-parameter -pedantic -Wdisabled-optimization -Wformat=2 -Winit-self -Wstrict-overflow=2 -Wswitch-default -fpermissive -std=gnu++11 -Wno-vla -Woverloaded-virtual -Wctor-dtor-privacy -Wsign-promo -Weffc++ -Wno-format-nonliteral -Wno-overlength-strings -Wno-strict-overflow -march=armv8-a -Wno-ignored-attributes -Werror -O3 -ftree-vectorize -D_GLIBCXX_USE_NANOSLEEP -DARM_COMPUTE_CPP_SCHEDULER=1 -DARM_COMPUTE_AARCH64_V8A -DNO_DOT_IN_TOOLCHAIN -DEMBEDDED_KERNELS -Iinclude -I. -I. examples/graph_alexnet_2.cpp
	g++ -o build/examples/graph_alexnet_2 -Wl,--allow-shlib-undefined build/examples/graph_alexnet_2.o build/utils/Utils.o build/utils/GraphUtils.o build/utils/CommonGraphOptions.o -Lbuild -L. -lpthread -larm_compute_graph -larm_compute -larm_compute_core
