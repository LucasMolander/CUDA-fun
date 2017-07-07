#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"

int getValue(Matrix *m, int r, int c)
{
    int offset;

    if (m->rowMajor) {
        offset = r * m->nCols;
        offset += c;
    } else {
        offset = c * m->nRows;
        offset += r;
    }

    return *(m->values + offset);
}

void setValue(Matrix *m, int r, int c, int value)
{
    int offset;

    if (m->rowMajor) {
        offset = r * m->nCols;
        offset += c;
    } else {
        offset = c * m->nRows;
        offset += r;
    }
    
    *(m->values + offset) = value;
}

void destroyMatrix(Matrix *m)
{
    free(m->values);
    free(m);
}

Matrix *generateMatrix(int nRows, int nCols, bool rowMajor)
{
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));

    int size = sizeof(int) * nRows * nCols;

    m->values = (int *) malloc(size);
    m->rowMajor = rowMajor;
    m->nRows = nRows;
    m->nCols = nCols;

    return m;
}

int dotProduct(int *v1, int *v2, int n)
{
    int total = 0;

    for (int i = 0; i < n; i++) {
        total += v1[i] * v2[i];
    }

    return total;
}

void printMatrix(Matrix *m)
{
    for (int r = 0; r < m->nRows; r++) {
        for (int c = 0; c < m->nCols; c++) {
            printf("%d", getValue(m, r, c));

            if (c + 1 < m->nCols) {
                printf(" ");
            }
        }

        printf("\n");
    }
}

void fillMatrixStepwise(Matrix *m)
{
    int val = 0;

    for (int r = 0; r < m->nRows; r++) {
        for (int c = 0; c < m->nCols; c++) {
            setValue(m, r, c, val);
            val++;
        }
    }
}