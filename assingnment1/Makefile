INC_DIR=include
CFLAGS=-g -O0 --gpu-architecture=sm_80 -m64
CC=nvcc
OBJ_DIR=build
SRC_DIR=src
OBJS= build/gpu.o build/time_utils.o build/mtx.o

all: build/main
build/main: $(OBJS)
	$(CC) $(CFLAGS) src/main.cpp $(OBJS) -o build/main

build/gpu.o:
	$(CC) $(CFLAGS) -c src/main.cu -o build/gpu.o

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@
