# create a makefile variable named CC for your C/C++ compiler (es gcc, c++, ... )

# create a makefile variable named OPT with your favorite C flags (at least with -std=c99 -O3)
CC=gcc

OPT=-g -O0 -Wall
INCLUDE=
CFLAGS=$(OPT) $(INCLUDE)


all: main.out

main.out: main.c Makefile include/mtx.h include/time_utils.h include/vec_matrix.h
	$(CC) $(CFLAGS) main.c -o main.out

clean:
	rm -f main.out