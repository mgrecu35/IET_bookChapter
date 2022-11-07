gcc -c -fPIC -I/Users/mgrecu/multiscatter-1.2.10/include multiscatter2_ascii.c
f2py -c -m multiscatterLib multiscatter.f90 multiscatter2_ascii.o -L/Users/mgrecu/multiscatter-1.2.10/lib -lmultiscatter
