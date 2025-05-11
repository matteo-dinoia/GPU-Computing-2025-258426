#ifndef MTX_H
#define MTX_H

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
bool read_mtx_header(FILE *, struct Coo *);

// Transpose matrix so that it is sorted like i want it
bool read_mtx_data(FILE *, const struct Coo *);

#endif //MTX_H
