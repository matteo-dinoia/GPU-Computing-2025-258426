# MMIO
DIST_MMIO=distributed_mmio
DIST_MMIO_INCLUDE=$(DIST_MMIO)/include
DIST_MMIO_OBJ=build/dmmio_mmio.o build/dmmio_utils.o

# CPP
CFLAGS=-Wall -Wextra -g
INC_DIR=src/include
OBJS=$(DIST_MMIO_OBJ) build/gpu.o build/tester.o build/time_utils.o build/utils.o

# CUDA
NVCC_FLAGS=--gpu-architecture=sm_80 -m64 --compiler-options "$(CFLAGS)" -O3
INCLUDES=-I$(INC_DIR) -I$(DIST_MMIO_INCLUDE)
PROFILE_FLAGS=-lineinfo

# GPU
all: build/main build/main_profiling
normal: build/main
profiling: build/main_profiling
build/main: src/main.cu $(OBJS) $(INC_DIR)  Makefile | build
	nvcc $(NVCC_FLAGS) $(INCLUDES) $< $(OBJS) -o $@

build/main_profiling: src/main.cu $(OBJS) $(INC_DIR) Makefile | build
	nvcc $(NVCC_FLAGS) $(PROFILE_FLAGS) $(NVCC_INCLUDES) $< $(OBJS) -o $@

build/dmmio_mmio.o: $(DIST_MMIO)/src/mmio.cpp
	nvcc $(NVCC_FLAGS) -c $< -o $@

build/dmmio_utils.o: $(DIST_MMIO)/src/mmio_utils.cpp
	nvcc $(NVCC_FLAGS) -c $< -o $@

build/%.o: src/%.cu $(INC_DIR) Makefile | build
	nvcc $(NVCC_FLAGS) -c $< -o $@

# TODO improve
build/mtx_to_bmtx: distributed_mmio/src/mtx_to_bmtx.cpp Makefile | build
	g++ $(CFLAGS) $< -o $@ distributed_mmio/src/mmio_utils.cpp distributed_mmio/src/mmio.cpp

build:
	mkdir -p build

clean:
	rm -f build/*

.PHONY: all clean normal profiling
# mt7601u 3-1:1.0: Vendor request req:07 off:0208 failed:-71
