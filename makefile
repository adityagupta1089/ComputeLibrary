all:
	scons -j8 -Q arch=arm64-v8a build=native neon=True opencl=True
