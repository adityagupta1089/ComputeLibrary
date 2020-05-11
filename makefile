all: hikey
hikey:
	make -f makefile.arm sched
macos:
	make -f makefile.macos
