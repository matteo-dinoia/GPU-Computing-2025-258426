# MMIO
DIST_MMIO_SRC=distributed_mmio/src/mmio.cpp distributed_mmio/src/mmio_utils.cpp


# CPP
CPP=g++
CFLAGS=-g -O3
INC_DIR=src/include
OBJS=build/time_utils.o build/utils.o

# CUDA
NVCC_FLAGS=$(CFLAGS) --gpu-architecture=sm_80 -m64
GPU_OBJS= build/gpu.cu.o build/tester.cu.o $(OBJS) build/mtx.o build/cpu.o

all: gpu cpu converter
converter: build/mtx_to_bmtx build/mtx_to_sbmtx

# CPU
cpu: build/main-cpu
build/main-cpu: src/main-cpu.cpp $(OBJS) $(INC_DIR) build/dmmio_mmio.o build/dmmio_utils.o | build
	$(CPP) $(CFLAGS)  $(OBJS) build/dmmio_mmio.o build/dmmio_utils.o $< -o $@

build/%.o: src/%.cpp $(INC_DIR) | build
	$(CPP) $(CFLAGS) -c $< -o $@

# GPU
gpu: build/main
build/main: src/main.cu $(GPU_OBJS) $(INC_DIR) | build
	nvcc $(NVCC_FLAGS) src/main.cu $(GPU_OBJS) -o build/main

build/%.cu.o: src/%.cu $(INC_DIR) | build
	nvcc $(NVCC_FLAGS) -c $< -o $@

# MTX
build/mtx_to_bmtx: distributed_mmio/src/mtx_to_bmtx.cpp build/dmmio_mmio.o build/dmmio_utils.o Makefile | build
	g++ $(DIST_MMIO_FLAGS) $< build/dmmio_mmio.o build/dmmio_utils.o -o $@

build/mtx_to_sbmtx: distributed_mmio/src/mtx_to_sbmtx.cpp build/dmmio_mmio.o build/dmmio_utils.o Makefile | build
	g++ $(DIST_MMIO_FLAGS) $< build/dmmio_mmio.o build/dmmio_utils.o -o $@

build/dmmio_mmio.o: distributed_mmio/src/mmio.cpp | build
	g++ $(DIST_MMIO_FLAGS) -c $< -o $@

build/dmmio_utils.o: distributed_mmio/src/mmio_utils.cpp | build
	g++ $(DIST_MMIO_FLAGS) -c $< -o $@


build:
	mkdir -p build

clean:
	rm -f build/*

.PHONY: all clean cpu gpu converter