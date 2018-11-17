This is a fork of the original library:
    - to be able to be run on **Hikey 970** board using a `makefile` on top of `scons` as `NEON` target
    - also contains modified examples which makes the graph run 100 times to average out the running times while I experiment with the various parameters
    - Also forces the `C++11` threads scheduler to use a work stealing based work distribution approach (`DYNAMIC`)
