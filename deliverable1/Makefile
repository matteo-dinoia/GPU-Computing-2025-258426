# CPP
CPP=g++
CFLAGS=-g -O0
INC_DIR=src/include
OBJS=build/time_utils.o build/mtx.o build/cpu.o build/utils.o

# CUDA
NVCC_FLAGS=$(CFLAGS) --gpu-architecture=sm_80 -m64
GPU_OBJS= build/gpu.cu.o  build/tester.cu.o $(OBJS)

all: gpu cpu

# CPU
cpu: build/main-cpu
build/main-cpu: src/main-cpu.cpp $(OBJS) $(INC_DIR) | build
	$(CPP) $(CFLAGS) src/main-cpu.cpp $(OBJS) -o build/main-cpu

build/%.o: src/%.cpp $(INC_DIR) | build
	$(CPP) $(CFLAGS) -c $< -o $@

# GPU
gpu: build/main
build/main: src/main.cu $(GPU_OBJS) $(INC_DIR) | build
	nvcc $(NVCC_FLAGS) src/main.cu $(GPU_OBJS) -o build/main

build/%.cu.o: src/%.cu $(INC_DIR) | build
	nvcc $(NVCC_FLAGS) -c $< -o $@

build:
	mkdir -p build

clean:
	rm -f build/*