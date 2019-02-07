default: debug
all: debug release
release:
	scons -j8 -Q build_dir=release arch=arm64-v8a build=native opencl=False neon=True
debug:
	scons -j8 -Q build_dir=debug arch=arm64-v8a build=native debug=True opencl=False neon=True
