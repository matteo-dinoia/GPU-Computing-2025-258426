#include <stdbool.h>
#include <stdio.h>
#include "include/mtx.h"

bool read_mtx_header(FILE *file, struct Coo *matrix) {
    if (matrix == NULL) {
        printf("Null pointer in read mtx header is null");
        return ERR;
    }

    // Skip any initial whitespace or comments
    int ch;
    while ((ch = fgetc(file)) == '%') {
        // Discards the entire line.
        while ((ch = fgetc(file)) != EOF && ch != '\n') {
            // Do nothing
        }
    }
    ungetc(ch, file);

    // Read the matrix dimensions and non-zero entries
    int read = fscanf(file, "%d %d %d", &matrix->COLS, &matrix->ROWS, &matrix->NON_ZERO);
    return read == 3;
}

// Transpose matrix so that it is sorted like i want it
bool read_mtx_data(FILE *file, const struct Coo *matrix) {
    if (matrix == NULL) {
        printf("Null pointer in read mtx data is null");
        return ERR;
    }

    int row, col;
    float value;
    int i = 0;

    char line[100];
    while (fgets(line, sizeof(line), file) || i < matrix->NON_ZERO) {
        if (sscanf(line, "%d %d %f", &col, &row, &value) == 3) {
            // Store the entry (adjust 1-based index to 0-based)
            matrix->xs[i] = col - 1;
            matrix->ys[i] = row - 1;
            matrix->vals[i] = value;
            i++;
        }
    }

    // TODO re-add len check
    return OK;
}