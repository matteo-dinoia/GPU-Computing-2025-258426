#include <iostream>
#include <stdbool.h>
#include <fstream>
#include "include/mtx.h"

#define BUF_LEN 100

using std::cout, std::endl;

bool read_mtx_header(std::ifstream &file, struct Coo *matrix) {
    if (matrix == NULL) {
        cout << "Null pointer in read mtx header is null" << endl;
        return ERR;
    }

    // Skip any initial whitespace or comments
    char ch;
    while ((ch = file.get()) == '%') {
        // Discards the entire line.
        while (((ch = file.get())) != EOF && ch != '\n') {
        }
    }
    file.unget();

    // Read the matrix dimensions and non-zero entries
    char buf[BUF_LEN];
    file.getline(buf, BUF_LEN);
    int read = sscanf(buf, "%d %d %d", &matrix->COLS, &matrix->ROWS, &matrix->NON_ZERO);
    return read == 3;
}

// Transpose matrix so that it is sorted like i want it
bool read_mtx_data(std::ifstream &file, const struct Coo *matrix) {
    if (matrix == NULL) {
        cout << "Null pointer in read mtx data is null" << endl;
        return ERR;
    }

    int row, col;
    float value;
    int read;

    char buf[BUF_LEN];
    for (int line = 0; line < matrix->NON_ZERO; line++) {
        file.getline(buf, BUF_LEN);
        read = sscanf(buf, "%d %d %f", &col, &row, &value);

        if (read < 2) {
            return ERR;
        } else if (read == 2) {
            value = 1;
        }

        // Store the entry (adjust 1-based index to 0-based)
        matrix->xs[line] = col - 1;
        matrix->ys[line] = row - 1;
        matrix->vals[line] = value;
    }

    return OK;
}