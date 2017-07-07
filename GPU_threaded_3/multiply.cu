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
 * NOTE: Both A and B are row-major!
 */
__global__ void doMultiply(int *d_result, int *d_a, int *d_b, int m, int n, int p)
{
    // int runLength = n;
    int runLength = n / TILE_SIZE;

    // A stays on its row, and B stays on its column
    int aBlockRow = blockIdx.y;
    int aRow = aBlockRow * TILE_SIZE;

    int bBlockCol = blockIdx.x;
    int bCol = bBlockCol * TILE_SIZE;

    // Sum for this thread
    int sum = 0;

    for (int i = 0; i < runLength; i++) {
        // Move A along its row and B along its column
        int aBlockCol = i;
        int bBlockRow = i;
        
        // Actual indices into the matrices
        int aCol = aBlockCol * TILE_SIZE;
        int bRow = bBlockRow * TILE_SIZE;

        __shared__ int aTile[TILE_SIZE][TILE_SIZE];
        __shared__ int bTile[TILE_SIZE][TILE_SIZE];

        // Have each thread fill one element
        aTile[threadIdx.y][threadIdx.x] = d_a[(aRow + threadIdx.y) * n + (aCol + threadIdx.x)];
        bTile[threadIdx.y][threadIdx.x] = d_b[(bRow + threadIdx.y) * p + (bCol + threadIdx.x)];

        // Don't calculate until moving global to shared is done
        __syncthreads();

        // Each thread calculates its dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        }

        // Don't mess with shared memory until we're done using it
        __syncthreads();
    }

    d_result[(blockIdx.y * TILE_SIZE + threadIdx.y) * p + (blockIdx.x * TILE_SIZE + threadIdx.x)] = sum;
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
    dim3 grid(p / TILE_SIZE, m / TILE_SIZE, 1); // Remember that p and m are multiples of TILE_SIZE
    dim3 block(TILE_SIZE, TILE_SIZE, 1);

    doMultiply<<<grid, block>>>(d_result, d_a, d_b, m, n, p);

    // doMultiply<<<grid, block>>>(d_result, d_a, d_b, m, n, p, blockMatrixRows_A, blockMatrixCols_A, blockMatrixRows_B, blockMatrixCols_B);

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
