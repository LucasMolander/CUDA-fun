#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

typedef struct Matrix {
    int nRows;
    int nCols;

    bool rowMajor;

    int *values;
} Matrix;

#endif