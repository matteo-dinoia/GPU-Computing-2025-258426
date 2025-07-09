# MMIO
DIST_MMIO=distributed_mmio
DIST_MMIO_INCLUDE=$(DIST_MMIO)/include
DIST_MMIO_OBJ=build/dmmio_mmio.o build/dmmio_utils.o

# CPP
CFLAGS=-Wall -Wextra
INC_DIR=src/include
OBJS=$(DIST_MMIO_OBJ) build/gpu.o build/tester.o build/time_utils.o build/utils.o build/parameters.o
PROF_OBJS=$(DIST_MMIO_OBJ) build/gpu_profiling.o build/tester_profiling.o build/time_utils_profiling.o build/utils_profiling.o build/parameters_profiling.o

# CUDA
NVCC_FLAGS=--gpu-architecture=sm_80 -m64 --compiler-options "$(CFLAGS)" -O3 -g
PROFILE_FLAGS=-lineinfo

# GPU
all: normal profiling converter
normal: build/main
profiling: build/main_profiling
converter: build/mtx_to_bmtx

build/main: src/main.cu $(OBJS) $(INC_DIR)  Makefile | build
	nvcc $(NVCC_FLAGS) $(INCLUDES) $< $(OBJS) -o $@

build/%.o: src/%.cu $(INC_DIR) Makefile | build
	nvcc $(NVCC_FLAGS) -c $< -o $@

build/main_profiling: src/main.cu $(PROF_OBJS) $(INC_DIR) Makefile | build
	nvcc $(NVCC_FLAGS) $(PROFILE_FLAGS) $< $(PROF_OBJS) -o $@

build/%_profiling.o: src/%.cu $(INC_DIR) Makefile | build
	nvcc $(NVCC_FLAGS) $(PROFILE_FLAGS) -c $< -o $@

# MTX
build/mtx_to_bmtx: distributed_mmio/src/mtx_to_bmtx.cpp build/dmmio_mmio.o build/dmmio_utils.o Makefile | build
	g++ $(CFLAGS) $< -o $@ build/dmmio_mmio.o build/dmmio_utils.o

build/dmmio_mmio.o: $(DIST_MMIO)/src/mmio.cpp
	nvcc $(NVCC_FLAGS) -c $< -o $@

build/dmmio_utils.o: $(DIST_MMIO)/src/mmio_utils.cpp
	nvcc $(NVCC_FLAGS) -c $< -o $@


build:
	mkdir -p build

clean:
	rm -f build/*

.PHONY: all clean normal profiling converter
# mt7601u 3-1:1.0: Vendor request req:07 off:0208 failed:-71
