#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <cuda_runtime.h>

#include "../matrix.c"
#include "../matrix.h"

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
 * Work for a specific element.
 * A thread may work on multiple of these.
 * 
 * The length of each vector needs to be known,
 * but that can be an argument to the kernel.
 */
typedef struct ElementWork {
    int *v1;            // Device
    int *v2;            // Device
    int *resultElement; // Device
} ElementWork;

__global__ void doMultiply(int nThreads, int nElements, ElementWork *work, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < nElements) {

        // Pre-calculated vectors and result element
        ElementWork currentWork = work[idx];
        int *v1 = currentWork.v1;
        int *v2 = currentWork.v2;
        int *resultElement = currentWork.resultElement;

        int dotProd = 0;
        for (int i = 0; i < n; i++) {
            dotProd += v1[i] * v2[i];
        }

        *resultElement = dotProd;

        idx += nThreads;
    }
}

// /**
//  * Work for a thread to do.
//  */
// typedef struct ThreadWork {
//     elementWork ElementWork[];
// } ThreadWork;

/**
 * Assumes a is row-major and b is column-major.
 */
Matrix *multiply(Matrix *a, Matrix *b, bool resultRowMajor)
{
    Matrix *result = generateMatrix(a->nRows, b->nCols, resultRowMajor);

    // (m x n) * (n x p)
    int m = a->nRows;
    int n = a->nCols;
    int p = b->nCols;

    cudaError_t rc;

    // Move A to device
    int *d_a = NULL;
    if ((rc = cudaMalloc((void **) &d_a, m * n * sizeof(int))) != cudaSuccess)
        failCuda("Failed to allocate space for A\n", rc);
        
    if ((rc = cudaMemcpy(d_a, a->values, m * n * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess)
        failCuda("Failed to copy A over\n", rc);

    // Move B to device
    int *d_b = NULL;
    if ((rc = cudaMalloc((void **) &d_b, n * p * sizeof(int))) != cudaSuccess)
        failCuda("Failed to allocate space for B\n", rc);
    
    if ((rc = cudaMemcpy(d_b, b->values, n * p * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess)
        failCuda("Failed to copy B over\n", rc);

    // Allocate space for AB
    int *d_result = NULL;
    if ((rc = cudaMalloc((void **)&d_result, m * p * sizeof(int))) != cudaSuccess)
        failCuda("Failed to allocate space for the result matrix\n", rc);

    // Calculate
    int threadsPerBlock = 192;
    // int nBlocks = 13;
    int nBlocks = 128;
    int nThreads = threadsPerBlock * nBlocks;
    int nElements = m * p;

    // Calculate args, positions, offsets, etc. before calling the Kernel
    ElementWork *elementWork = (ElementWork *) malloc(nElements * sizeof(ElementWork));
    for (int i = 0; i < nElements; i++) {
        int r = i / p; // Integer division on purpose
        int c = i % p;

        int *v1 = d_a + (r * n);
        int *v2 = d_b + (c * n);

        int offset;
        if (resultRowMajor) {
            offset = r * p;
            offset += c;
        } else {
            offset = c * m;
            offset += r;
        }
        int *resultElement = d_result + offset;

        elementWork[i].v1 = v1;
        elementWork[i].v2 = v2;
        elementWork[i].resultElement = resultElement;
    }

    // Copy the element work over there so the kernels have access to it
    ElementWork *d_elementWork = NULL;
    if ((rc = cudaMalloc((void **) &d_elementWork, nElements * sizeof(ElementWork))) != cudaSuccess)
        failCuda("Failed to allocate space for element work\n", rc);
    
    if ((rc = cudaMemcpy(d_elementWork, elementWork, nElements * sizeof(ElementWork), cudaMemcpyHostToDevice)) != cudaSuccess)
        failCuda("Failed to copy element work over\n", rc);

    // Send the threads to the kernel...
    doMultiply<<<nBlocks, threadsPerBlock>>>(nThreads, nElements, d_elementWork, n);

    // Make sure nothing bad happened
    if ((rc = cudaGetLastError()) != cudaSuccess)
        failCuda("Failure in CUDA kernel execution\n", rc);
    
    // Copy the result matrix's values back to host
    if ((rc = cudaMemcpy(result->values, d_result, nElements * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess)
        failCuda("Failed to copy result matrix to host\n", rc);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaFree(d_elementWork);

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
    // printf("\n");

    // Clean up
    destroyMatrix(a);
    destroyMatrix(b);
    destroyMatrix(ab);
    
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
