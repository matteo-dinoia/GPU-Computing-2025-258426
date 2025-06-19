#pragma once

#define OK true
#define ERR false

#include <iostream>

struct Coo {
    int NON_ZERO;
    int ROWS;
    int COLS;
    int *xs;
    int *ys;
    float *vals;
};

// Function to skip comments and read metadata from the MTX file
// Transpose matrix so that it is sorted like i want it
bool read_mtx_header(std::ifstream &file, struct Coo *);

// Transpose matrix so that it is sorted like i want it
bool read_mtx_data(std::ifstream &file, const struct Coo *);
