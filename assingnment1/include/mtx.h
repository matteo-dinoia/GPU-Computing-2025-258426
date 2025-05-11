#ifndef MTX_H
#define MTX_H

#define M_TYPE float
#include <stdbool.h>
#include <stdio.h>

// Function to skip comments and read metadata from the MTX file
// Transpose matrix so that it is sorted like i want it
bool read_mtx_header(FILE *file, int *rows, int *cols, int *non_zeros) {
    // Skip any initial whitespace or comments
    int ch;
    while ((ch = fgetc(file)) == '%') {
        // Discards the entire line.
        while ((ch = fgetc(file)) != EOF && ch != '\n')
            ;
    }
    ungetc(ch, file);

    // Read the matrix dimensions and non-zero entries
    int read = fscanf(file, "%d %d %d", cols, rows, non_zeros);
    return read == 3;
}

// Transpose matrix so that it is sorted like i want it
bool read_mtx_data(FILE *file, int *coords_x, int *coords_y, M_TYPE *vals, const int LEN) {
    if (coords_x == NULL || coords_y == NULL || vals == NULL) {
        printf("Null pointer in read mtx data is null");
        return false;
    }

    int row, col;
    M_TYPE value;
    int i = 0;

    char line[100];
    while (fgets(line, sizeof(line), file) || i < LEN) {
        if (sscanf(line, "%d %d %f", &col, &row, &value) == 3) {
            // Store the entry (adjust 1-based index to 0-based)
            coords_x[i] = col - 1;
            coords_y[i] = row - 1;
            vals[i] = value;
            i++;
        }
    }

    /*while (fscanf(file, "%d %d %f", &col, &row, &value) == 3) {
        // Store the entry (adjust 1-based index to 0-based)
        coords_x[i] = col - 1;
        coords_y[i] = row - 1;
        vals[i] = value;
        i++;
    }

    if (i != LEN) {
        printf("ERROR: didn't read the right amount of data\n");
        printf("INFO: read %d of %d data line\n", i, LEN);
        return false;
    } TODO READD */
    return true;
}

#endif //MTX_H
