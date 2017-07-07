/**
 * This was a failed attempt at dynamically tiling the input matrices.
 */

#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>

#include "../matrix.c"
#include "../matrix.h"

#define TILE_SIZE (32)

void fail(const char *message)
{
    printf(message);
    fflush(stdout);
    exit(EXIT_FAILURE);
}

void failCuda(const char *message, cudaError_t rc)
{
    printf(message);
    printf("CUDA error: %s\n", cudaGetErrorString(rc));
    fflush(stdout);
    exit(EXIT_FAILURE);
}

/**
 * Helper for moving workers through a matrix of work.
 * 
 * If the next index is out side of the matrix (work is done),
 * then it sets currentRow and currentCol to -1.
 */
__device__ void getNextIndices(int *currentRow, int *currentCol, int nRows, int nCols, int nWorkers)
{
    int currentIndex = ((*currentRow) * nCols) + (*currentCol);

    int nextIndex = currentIndex + nWorkers;

    int nWorkToDo = nRows * nCols;

    if (nextIndex >= nWorkToDo) { // Done working
        *currentRow = -1;
        *currentCol = -1;
    } else {
        *currentRow = nextIndex / nCols; // Integer division on purpose
        *currentCol = nextIndex % nCols;
    }
}

__device__ void printSubMatrix(int *matrix, int nRows, int nCols, int beginR, int beginC, int subRows, int subCols)
{
    for (int r = 0; r < subRows; r++) {
        for (int c = 0; c < subCols; c++) {

            int val = matrix[(beginR + r) * nCols + (beginC + c)];

            if (val <= 9 && val >= 0) {
                printf("  ");
            } else if (val <= 99 && val >= 10) {
                printf(" ");
            }

            printf("%d", val);

            if (c + 1 < subCols) {
                printf(" ");
            }
        }

        printf("\n");
    }
}

/**
 * NOTE: Both A and B are row-major!
 */
__global__ void doMultiply(int *d_result, int *d_a, int *d_b, int m, int n, int p,
    int blockMatrixRows_A, int blockMatrixCols_A, int blockMatrixRows_B, int blockMatrixCols_B)
{
    __shared__ int aTile[TILE_SIZE][TILE_SIZE];
    __shared__ int bTile[TILE_SIZE][TILE_SIZE];

    int nBlocks = gridDim.x;

    int nBlockMatrixElements_A = blockMatrixRows_A * blockMatrixCols_A;
    // int nBlockMatrixElements_B = blockMatrixRows_B * blockMatrixCols_B;
    
    // No work for this block
    if (blockIdx.x >= nBlockMatrixElements_A) {
        return;
    }

    // Calculate initial block matrix indices
    int blockMatrixRow_A = blockIdx.x / blockMatrixCols_A;
    int blockMatrixCol_A = blockIdx.x % blockMatrixCols_A;

    // While there's work for this block
    while (blockMatrixRow_A != -1 && blockMatrixCol_A != -1) {

        // Copy tile of A to shared memory
        int aRow = blockMatrixRow_A * TILE_SIZE;
        int aCol = blockMatrixCol_A * TILE_SIZE;

        int aRowsLeft = m - aRow;
        int aColsLeft = n - aCol;
        int tileM_A = (aRowsLeft < TILE_SIZE) ? aRowsLeft : TILE_SIZE;
        int tileN_A = (aColsLeft < TILE_SIZE) ? aColsLeft : TILE_SIZE;

        for (int r = 0; r < tileM_A; r++) {
            for (int c = 0; c < tileN_A; c++) {
                aTile[r][c] = d_a[(aRow + r) * n + (aCol + c)];
            }
        }

        // While we're moving along B
        int blockMatrixRow_B = blockMatrixCol_A;
        int blockMatrixCol_B = 0;
        while (blockMatrixCol_B != -1) {
            // Copy tile of B to shared memory
            int bRow = blockMatrixRow_B * TILE_SIZE;
            int bCol = blockMatrixCol_B * TILE_SIZE;

            int bRowsLeft = n - bRow;
            int bColsLeft = p - bCol;
            int tileM_B = (bRowsLeft < TILE_SIZE) ? bRowsLeft : TILE_SIZE;
            int tileN_B = (bColsLeft < TILE_SIZE) ? bColsLeft : TILE_SIZE;

            for (int r = 0; r < tileM_B; r++) {
                for (int c = 0; c < tileN_B; c++) {
                    int totalOffset = (bRow + r) * p + (bCol + c);
                    bTile[r][c] = d_b[totalOffset];
                }
            }

            __syncthreads();

            int runLength = tileN_A;

            // The thread's row and column need to be within the constraints
            if (threadIdx.y < tileM_A && threadIdx.x < tileN_B) {
                for (int i = 0; i < runLength; i++) {
                    // if (blockIdx.x == 6) {
                    //     printf("\t[%d (+%d)][%d (+%d)] += [%d][%d] * [%d][%d]\n", threadIdx.y, aRow, threadIdx.x, bCol, threadIdx.y, i, i, threadIdx.x);
                    // }

                    d_result[(aRow + threadIdx.y) * p + (bCol + threadIdx.x)] += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
                }
            }

            blockMatrixCol_B++;
            if (blockMatrixCol_B >= blockMatrixCols_B) {
                blockMatrixCol_B = -1;
            }
        }

        getNextIndices(&blockMatrixRow_A, &blockMatrixCol_A, blockMatrixRows_A, blockMatrixCols_A, nBlocks);
    }
}

/**
 * NOTE: Both A and B are row-major!
 */
Matrix *multiply(Matrix *a, Matrix *b, bool resultRowMajor)
{
    Matrix *result = generateMatrix(a->nRows, b->nCols, resultRowMajor);

    // (m x n) * (n x p)
    int m = a->nRows;
    int n = a->nCols;
    int p = b->nCols;

    int aElements = m * n;
    int bElements = n * p;
    int resultElements = m * p;

    cudaError_t rc;

    // Move A to device
    int *d_a = NULL;
    if ((rc = cudaMalloc((void **) &d_a, aElements * sizeof(int))) != cudaSuccess)
        failCuda("Failed to allocate space for A\n", rc);
        
    if ((rc = cudaMemcpy(d_a, a->values, aElements * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess)
        failCuda("Failed to copy A over\n", rc);

    // Move B to device
    int *d_b = NULL;
    if ((rc = cudaMalloc((void **) &d_b, bElements * sizeof(int))) != cudaSuccess)
        failCuda("Failed to allocate space for B\n", rc);
    
    if ((rc = cudaMemcpy(d_b, b->values, bElements * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess)
        failCuda("Failed to copy B over\n", rc);

    // Allocate space for AB
    int *d_result = NULL;
    if ((rc = cudaMalloc((void **)&d_result, resultElements * sizeof(int))) != cudaSuccess)
        failCuda("Failed to allocate space for the result matrix\n", rc);

    // Zero it out because doMultiply adds to it piecemeal
    if ((rc = cudaMemset(d_result, 0, resultElements * sizeof(int))) != cudaSuccess)
        failCuda("Failed to zero-out result matrix\n", rc);

    // Send the threads to the kernel...
    dim3 grid(32, 1, 1);
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    // int nThreads = (grid.x * grid.y * grid.z) * (block.x * block.y * block.z);

    int blockMatrixRows_A = (int) ceil((double) m / (double) TILE_SIZE);
    int blockMatrixCols_A = (int) ceil((double) n / (double) TILE_SIZE);
    int blockMatrixRows_B = (int) ceil((double) n / (double) TILE_SIZE);
    int blockMatrixCols_B = (int) ceil((double) p / (double) TILE_SIZE);

    doMultiply<<<grid, block>>>(d_result, d_a, d_b, m, n, p, blockMatrixRows_A, blockMatrixCols_A, blockMatrixRows_B, blockMatrixCols_B);

    // Make sure nothing bad happened
    if ((rc = cudaGetLastError()) != cudaSuccess)
        failCuda("Failure in CUDA kernel execution\n", rc);
    
    // Copy the result matrix's values back to host
    if ((rc = cudaMemcpy(result->values, d_result, resultElements * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess)
        failCuda("Failed to copy result matrix to host\n", rc);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

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
    // printf("Tile size: %d\n", TILE_SIZE);
    // printf("\n");

    Matrix *a = generateMatrix(nRows1, nCols1, true);
    fillMatrixStepwise(a);
    
    Matrix *b = generateMatrix(nRows2, nCols2, true);
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
