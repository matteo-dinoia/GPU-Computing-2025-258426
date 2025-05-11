#ifndef MTX_H
#define MTX_H

#define M_TYPE float
#include <stdbool.h>
#include <stdio.h>

// Function to skip comments and read metadata from the MTX file
// Transpose matrix so that it is sorted like i want it
bool read_mtx_header(FILE *file, int *rows, int *cols, int *non_zeros) {
    int ch;

    // Skip any initial whitespace or comments
    while ((ch = fgetc(file)) == '%') {
        while ((ch = fgetc(file)) != EOF && ch != '\n')
            ; // Discards the entire line.
    }
    ungetc(ch, file);

    // Read the matrix dimensions and non-zero entries
    if (fscanf(file, "%d %d %d", cols, rows, non_zeros) != 3) {
        return false;
    }

    return true;
}

// Transpose matrix so that it is sorted like i want it
bool read_mtx_data(FILE *file, int *coords_x, int *coords_y, M_TYPE *vals, const int LEN) {
    if (coords_x == NULL || coords_y == NULL || vals == NULL) {
        printf("Null pointer in read mtx data is null");
        return false;
    }
    int row, col;
    double value;
    int i = 0;

    while (fscanf(file, "%d %d %lf", &col, &row, &value) == 3) {
        // Store the entry (adjust 1-based index to 0-based)
        coords_x[i] = col - 1;
        coords_y[i] = row - 1;
        vals[i] = value;
        i++;
    }

    return i == LEN;
}

#endif //MTX_H
