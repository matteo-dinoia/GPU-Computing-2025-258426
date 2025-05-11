# create a makefile variable named CC for your C/C++ compiler (es gcc, c++, ... )

# create a makefile variable named OPT with your favorite C flags (at least with -std=c99 -O3)

BLAS_LIB=/opt/shares/openfoam/software/OpenBLAS/0.3.23-GCC-12.3.0
CC=gcc

OPT=-g -O0
INCLUDE=-lm
CFLAGS=$(OPT) $(INCLUDE)


all: main

main: main.c Makefile
	$(CC) $(CFLAGS) main.c -o main.out

clean:
	rm -f main.out