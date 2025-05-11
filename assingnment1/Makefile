INC_DIR=include
CFLAGS=-g -O0 --gpu-architecture=sm_80 -m64
CC=nvcc
OBJS= build/gpu.o build/time_utils.o build/mtx.o build/cpu.o

all: build/main
build/main: $(OBJS)
	$(CC) $(CFLAGS) src/main.cu $(OBJS) -o build/main

build/gpu.o:
	$(CC) $(CFLAGS) -c src/gpu.cu -o build/gpu.o

build/%.o: src/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -f build/*