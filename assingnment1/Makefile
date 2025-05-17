INC_DIR=include
CFLAGS=-g -O0 --gpu-architecture=sm_80 -m64
CC=nvcc
OBJS= build/gpu.cu.o build/time_utils.o build/mtx.o build/cpu.o build/tester.cu.o

all: build/main
build/main: $(OBJS) src/main.cu | build
	$(CC) $(CFLAGS) src/main.cu $(OBJS) -o build/main

build/%.cu.o: src/%.cu | build
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: src/%.cpp | build
	$(CC) $(CFLAGS) -c $< -o $@

build:
	mkdir -p build

clean:
	rm -f build/*