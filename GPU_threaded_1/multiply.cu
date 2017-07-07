/**
 * This ended up not being as efficient as CPU threads.
 * 
 * See the "Results 1" results for the benchmark data.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <cuda_runtime.h>

#include "../matrix.c"
#include "../matrix.h"

void fail(const char *message)
{
    printf(message);
    exit(EXIT_FAILURE);
}

__global__ void doMultiply(int *d_result, int *d_a, int *d_b, 
    int m, int n, int p, int nThreads)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int nElements = m * p;

    while (idx < nElements) {
        int r = idx / p; // Integer division on purpose
        int c = idx % p;

        int *v1 = d_a + (r * n);
        int *v2 = d_b + (c * n);

        int dotProd = 0;
        for (int i = 0; i < n; i++) {
            dotProd += v1[i] * v2[i];
        }

        d_result[r * p + c] = dotProd;

        idx += nThreads;
    }
}

/**
 * Assumes a is row-major and b is column-major.
 * 
 * Result is always row-major.
 */
Matrix *multiply(Matrix *a, Matrix *b)
{
    Matrix *result = generateMatrix(a->nRows, b->nCols, true);

    // Move A to device
    int *d_a = NULL;
    if (cudaMalloc((void **) &d_a, a->nRows * a->nCols * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for A");
        
    if (cudaMemcpy(d_a, a->values, a->nRows * a->nCols * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        fail("Failed to copy A over");

    // Move B to device
    int *d_b = NULL;
    if (cudaMalloc((void **) &d_b, b->nRows * b->nCols * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for B");
    
    if (cudaMemcpy(d_b, b->values, b->nRows * b->nCols * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        fail("Failed to copy B over");

    // Allocate space for AB
    int *d_result = NULL;
    if (cudaMalloc((void **)&d_result, a->nRows * b->nCols * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for the result matrix");

    // Calculate
    // int threadsPerBlock = 192;
    // int nBlocks = 13;
    int threadsPerBlock = 128;
    int nBlocks = 32;
    int nThreads = threadsPerBlock * nBlocks;
    int nElements = a->nRows * b->nCols;
    
    doMultiply<<<nBlocks, threadsPerBlock>>>(d_result, d_a, d_b, a->nRows, a->nCols, b->nCols, nThreads);
    if (cudaGetLastError() != cudaSuccess)
        fail("Failure in CUDA kernel execution");
    
    if (cudaMemcpy(result->values, d_result, nElements * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy result matrix to host");

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
    // printf("\n");

    Matrix *a = generateMatrix(nRows1, nCols1, true);
    fillMatrixStepwise(a);
    
    Matrix *b = generateMatrix(nRows2, nCols2, false);
    fillMatrixStepwise(b);

    Matrix *ab = multiply(a, b);

    // printMatrix(a);
    // printf("\n");
    // printMatrix(b);
    // printf("\n");
    // printMatrix(ab);

    // Clean up
    destroyMatrix(a);
    destroyMatrix(b);
    destroyMatrix(ab);
    
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
