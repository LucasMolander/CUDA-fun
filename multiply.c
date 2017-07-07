#include "matrix.c"
#include "matrix.h"

void fail(char *message)
{
    printf(message);
    exit(EXIT_FAILURE);
}

/**
 * Assumes a is row-major and b is column-major.
 */
Matrix *multiply(Matrix *a, Matrix *b, bool resultRowMajor)
{
    Matrix *result = generateMatrix(a->nRows, b->nCols, resultRowMajor);

    int vectorLength = a->nCols;

    for (int r = 0; r < a->nRows; r++) {        // r-th row for A
        int *v1 = a->values + (r * a->nCols);
        for (int c = 0; c < b->nCols; c++) {    // c-th col for B
            int *v2 = b->values + (c * b->nRows);
            setValue(result, r, c, dotProduct(v1, v2, vectorLength));
        }
    }

    return result;
}

int main(int argc, char *argv[])
{
    // Ensure enough arguments exist
    if (argc < 5)
        fail("Required arguments: nRows1 nCols1 nRows2 nCols2\n");

    int nRows1,
        nCols1,
        nRows2,
        nCols2;

    // It's okay that atoi returns 0 on invalid
    // because 0 is an invalid matrix dimension
    if ((nRows1 = atoi(argv[1])) == 0)
        fail("Invalid matrix dimension.\n");
    if ((nCols1 = atoi(argv[2])) == 0)
        fail("Invalid matrix dimension.\n");
    if ((nRows2 = atoi(argv[3])) == 0)
        fail("Invalid matrix dimension.\n");
    if ((nCols2 = atoi(argv[4])) == 0)
        fail("Invalid matrix dimension.\n");

    // Negative matrix dimensions are also bad
    if (nRows1 < 0 || nCols1 < 0 || nRows2 < 0 || nCols2 < 0)
        fail("Invalid matrix dimension.\n");
    
    // Make sure the matrix multiplication is valid
    if (nCols1 != nRows2)
        fail("Matrices cannot be multiplied (nCols1 needs to equal nRows2)\n");

    // Echo matrix dimensions to the user
    // printf("%d x %d\n", nRows1, nCols1);
    // printf("%d x %d\n", nRows2, nCols2);

    Matrix *a = generateMatrix(nRows1, nCols1, true);
    fillMatrixStepwise(a);
    
    Matrix *b = generateMatrix(nRows2, nCols2, false);
    fillMatrixStepwise(b);

    Matrix *ab = multiply(a, b, true);


    // printf("\n");
    // printMatrix(a);
    // printf("\n");
    // printMatrix(b);
    // printf("\n");
    // printMatrix(ab);

    destroyMatrix(a);
    destroyMatrix(b);
    destroyMatrix(ab);

    return EXIT_SUCCESS;
}
