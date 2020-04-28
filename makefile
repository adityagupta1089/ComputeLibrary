all: hikey
hikey:
	make -f makefile.debian sched
macos:
	make -f makefile.macos
