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
    bool resultRowMajor, int m, int n, int p)
{
    extern __shared__ int s[];

    int nBlocks = gridDim.x;
    int nThreadsPerBlock = blockDim.x;

    int sMemBytes = 114688; // 112 KB
    int sPerBlock = sMemBytes / nBlocks; // 3584
    int sElemPerBlock = sPerBlock / sizeof(int); // 896
    int sALength = sElemPerBlock / 2; // 448
    int sBLength = sElemPerBlock / 2; // 448
    int *sA = s;
    int *sB = (int *)&s[sALength];

    int nAElements = m * n;
    int nBElements = n * p;

    int aPerBlock = nAElements / nBlocks;
    int bPerBlock = nBElements / nBlocks;
    int aExtra = nAElements % nBlocks;
    int bExtra = nBElements % nBlocks;

    // Calculate offset into matrix A and length of A for this block
    int aAbove = (blockIdx.x > aExtra) ? blockIdx.x - aExtra : 0;
    int aBelow = (blockIdx.x > aAbove) ? blockIdx.x - aAbove : 0;
    int aOffset = ((aPerBlock + 1) * aBelow) + (aPerBlock * aAbove);
    int blockALength = aPerBlock + ((blockIdx.x < aExtra) ? 1 : 0);

    // Calculate offset into matrix B and length of B for this block
    int bAbove = (blockIdx.x > bExtra) ? blockIdx.x - bExtra : 0;
    int bBelow = (blockIdx.x > bAbove) ? blockIdx.x - bAbove : 0;
    int bOffset = ((bPerBlock + 1) * bBelow) + (bPerBlock * bAbove);
    int blockBLength = bPerBlock + ((blockIdx.x < bExtra) ? 1 : 0);

    // Make sure to also use aIndex and bIndex when indexing into d_a and d_b
    // for copying them to shared memory
    int *blockA = d_a + aOffset;
    int *blockB = d_b + bOffset;

    int aIndex = 0;
    int bIndex = 0;

    int vLength = n;
    int aDone = 0;
    int bDone = 0;

    while (aIndex < blockALength) {
        int aTotalIndex = aOffset + aIndex;
        int aNextRowBreak = aTotalIndex + (n - (aTotalIndex % n));

        // Minimum of:
        //  - Length to end of the block
        //  - Length to the end of the row
        //  - Length of shared memory size
        int aToEndOfBlock blockALength - aIndex;
        int aToRowBreak = aNextRowBreak - aTotalIndex;
        int nAToGet = (aToEndOfBlock < aToRowBreak) ? aToEndOfBlock : aToRowBreak;
        nAToGet = (nAToGet < sALength) ? nAToGet : sALength;

        // Copy into shared memory
        for (int i = 0; i < nAToGet; i++) {
            sA[i] = d_a[aTotalIndex + i];
        }

        // Make sure all threads have the shared data
        __syncThreads();

        aIndex += nAToGet;
    }
}

/**
 * Assumes a is row-major and b is column-major.
 */
Matrix *multiply(Matrix *a, Matrix *b, bool resultRowMajor)
{
    Matrix *result = generateMatrix(a->nRows, b->nCols, resultRowMajor);

    int m = a->nRows;
    int n = a->nCols; // Also equals b->nRows
    int p = b->nCols;

    // Move A to device
    int *d_a = NULL;
    if (cudaMalloc((void **) &d_a, m * n * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for A");
        
    if (cudaMemcpy(d_a, a->values, m * n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        fail("Failed to copy A over");

    // Move B to device
    int *d_b = NULL;
    if (cudaMalloc((void **) &d_b, n * p * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for B");
    
    if (cudaMemcpy(d_b, b->values, n * p * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        fail("Failed to copy B over");

    // Allocate space for AB
    int *d_result = NULL;
    if (cudaMalloc((void **)&d_result, m * p * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for the result matrix");

    // Calculate
    // int threadsPerBlock = 128;
    // int nBlocks = 32;
    int threadsPerBlock = 128;
    int nBlocks = 32;
    int nThreads = threadsPerBlock * nBlocks;
    int nElements = m * p;

    int sMemBytes = 114688; // 112 KB
    int sPerBlock = sMemBytes / nBlocks; // 3584
    int sElemPerBlock = sPerBlock / sizeof(int); // 896
    int sA = sElemPerBlock / 2; // 448
    int sB = sElemPerBlock / 2; // 448
    
    doMultiply<<<nBlocks, threadsPerBlock>>>(d_result, d_a, d_b, resultRowMajor, m, n, p, nThreads);

    // doMultiply<<<nBlocks, threadsPerBlock>>>(d_result, d_a, d_b, resultRowMajor, m, n, p, nThreads);
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

    Matrix *ab = multiply(a, b, true);

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