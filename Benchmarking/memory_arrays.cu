#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#ifndef N
#define N (1024)
#endif

void fail(const char *message)
{
    printf(message);
    exit(EXIT_FAILURE);
}

__global__ void localToLocal(unsigned long long *d_time)
{
    int arr1[N];
    int arr2[N];
    for (int i = 0; i < N; i++) {
        arr1[i] = i * 2 + 1;
        arr2[i] = i * 3 - 1;
    }

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        arr1[i] = arr2[i];
    }
    unsigned long long endTime = clock();

    // Use the local array so the compiler doesn't optimize it away
    arr1[0] = arr1[1];

    *d_time = (endTime - startTime);
}

__global__ void globalToGlobal(int *d_v1, int *d_v2, unsigned long long *d_time)
{
    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        d_v1[i] = d_v2[i];
    }
    unsigned long long endTime = clock();

    *d_time = (endTime - startTime);
}

__global__ void sharedToShared(unsigned long long *d_time)
{
    __shared__ int shared1[N];
    __shared__ int shared2[N];
    for (int i = 0; i < N; i++) {
        shared1[i] = i * 2 + 1;
        shared2[i] = i * 3 - 1;
    }

    __syncthreads();

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        shared1[i] = shared2[i];
    }
    unsigned long long endTime = clock();

    // Used the shared array so the compiler doesn't optimize it away
    shared1[0] = shared1[1];

    *d_time = (endTime - startTime);
}

__global__ void localToGlobal(int *d_v, unsigned long long *d_time)
{
    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        d_v[i] = i;
    }
    unsigned long long endTime = clock();

    *d_time = (endTime - startTime);
}

__global__ void globalToLocal(int *d_v, unsigned long long *d_time)
{
    int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i * 2 + 1;
    }

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        arr[i] = d_v[i];
    }
    unsigned long long endTime = clock();

    // Use the local array so the compiler doesn't optimize it away
    arr[0] = arr[1];

    *d_time = (endTime - startTime);
}

__global__ void sharedToGlobal(int *d_v, unsigned long long *d_time)
{
    __shared__ int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i * 2 + 1;
    }

    __syncthreads();

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        d_v[i] = arr[i];
    }
    unsigned long long endTime = clock();

    *d_time = (endTime - startTime);
}

__global__ void globalToShared(int *d_v, unsigned long long *d_time)
{
    __shared__ int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i * 2 + 1;
    }

    __syncthreads();

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        arr[i] = d_v[i];
    }
    unsigned long long endTime = clock();

    // Get rid of compiler warning
    arr[0] = arr[1];

    *d_time = (endTime - startTime);
}

__global__ void localToShared(unsigned long long *d_time)
{
    __shared__ int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i * 2 + 1;
    }

    int localArr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i * 3 - 1;
    }

    __syncthreads();

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        arr[i] = localArr[i];
    }
    unsigned long long endTime = clock();

    // Get rid of compiler warning
    arr[0] = arr[1];

    *d_time = (endTime - startTime);
}

__global__ void sharedToLocal(unsigned long long *d_time)
{
    __shared__ int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i * 2 + 1;
    }

    int localArr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i * 3 - 1;
    }

    __syncthreads();

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        localArr[i] = arr[i];
    }
    unsigned long long endTime = clock();

    // Use the local array so the compiler doesn't optimize it away
    localArr[0] = localArr[1];
    
    *d_time = (endTime - startTime);
}

int main()
{
    /**
     * Set up memory on device.
     */
    int *d_globalToGlobal1 = NULL;
    int *d_globalToGlobal2 = NULL;
    int *d_localToGlobal = NULL;
    int *d_globalToLocal = NULL;
    int *d_sharedToGlobal = NULL;
    int *d_globalToShared = NULL;
    if (cudaMalloc((void **) &d_globalToGlobal1, N * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for 'd_globalToGlobal1'");
    if (cudaMalloc((void **) &d_globalToGlobal2, N * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for 'd_globalToGlobal2'");
    if (cudaMalloc((void **) &d_localToGlobal, N * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for 'd_localToGlobal'");
    if (cudaMalloc((void **) &d_globalToLocal , N * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for 'd_globalToLocal'");
    if (cudaMalloc((void **) &d_sharedToGlobal , N * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for 'd_sharedToGlobal'");
    if (cudaMalloc((void **) &d_globalToShared , N * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for 'd_globalToShared'");

    unsigned long long tLocalToLocal;
    unsigned long long tGlobalToGlobal;
    unsigned long long tSharedToShared;
    unsigned long long tLocalToGLobal;
    unsigned long long tglobalToLocal;
    unsigned long long tsharedToGlobal;
    unsigned long long tglobalToShared;
    unsigned long long tlocalToShared;
    unsigned long long tsharedToLocal;

    unsigned long long *d_tLocalToLocal = NULL;
    unsigned long long *d_tGlobalToGlobal = NULL;
    unsigned long long *d_tSharedToShared = NULL;
    unsigned long long *d_tLocalToGLobal = NULL;
    unsigned long long *d_tglobalToLocal = NULL;
    unsigned long long *d_tsharedToGlobal = NULL;
    unsigned long long *d_tglobalToShared = NULL;
    unsigned long long *d_tlocalToShared = NULL;
    unsigned long long *d_tsharedToLocal = NULL;
    if (cudaMalloc((void **) &d_tLocalToLocal , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tLocalToLocal'");
    if (cudaMalloc((void **) &d_tGlobalToGlobal , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tGlobalToGlobal'");
    if (cudaMalloc((void **) &d_tSharedToShared , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tSharedToShared'");
    if (cudaMalloc((void **) &d_tLocalToGLobal , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tLocalToGLobal'");
    if (cudaMalloc((void **) &d_tglobalToLocal , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tglobalToLocal'");
    if (cudaMalloc((void **) &d_tsharedToGlobal , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tsharedToGlobal'");
    if (cudaMalloc((void **) &d_tglobalToShared , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tglobalToShared'");
    if (cudaMalloc((void **) &d_tlocalToShared , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tlocalToShared'");
    if (cudaMalloc((void **) &d_tsharedToLocal , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tsharedToLocal'");



    /**
     * Execute kernels.
     */
    int nBlocks = 32;
    int nThreads = 128;

    localToLocal<<<nBlocks, nThreads>>>(d_tLocalToLocal);
    globalToGlobal<<<nBlocks, nThreads>>>(d_globalToGlobal1, d_globalToGlobal2, d_tGlobalToGlobal);
    sharedToShared<<<nBlocks, nThreads>>>(d_tSharedToShared);
    localToGlobal<<<nBlocks, nThreads>>>(d_localToGlobal, d_tLocalToGLobal);
    globalToLocal<<<nBlocks, nThreads>>>(d_globalToLocal, d_tglobalToLocal);
    sharedToGlobal<<<nBlocks, nThreads>>>(d_sharedToGlobal, d_tsharedToGlobal);
    globalToShared<<<nBlocks, nThreads>>>(d_globalToShared, d_tglobalToShared);
    localToShared<<<nBlocks, nThreads>>>(d_tlocalToShared);
    sharedToLocal<<<nBlocks, nThreads>>>(d_tsharedToLocal);



    /**
     * Copy results back.
     */
    if (cudaMemcpy(&tLocalToLocal, d_tLocalToLocal, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tLocalToLocal");
    if (cudaMemcpy(&tGlobalToGlobal, d_tGlobalToGlobal, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tGlobalToGlobal");
    if (cudaMemcpy(&tSharedToShared, d_tSharedToShared, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tSharedToShared");
    if (cudaMemcpy(&tLocalToGLobal, d_tLocalToGLobal, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tLocalToGLobal");
    if (cudaMemcpy(&tglobalToLocal, d_tglobalToLocal, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tglobalToLocal");
    if (cudaMemcpy(&tsharedToGlobal, d_tsharedToGlobal, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tsharedToGlobal");
    if (cudaMemcpy(&tglobalToShared, d_tglobalToShared, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tglobalToShared");
    if (cudaMemcpy(&tlocalToShared, d_tlocalToShared, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tlocalToShared");
    if (cudaMemcpy(&tsharedToLocal, d_tsharedToLocal, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tsharedToLocal");



    /**
     * Print results.
     */
    printf("Benchmark: moving ints between arrays\n");
    printf("N = %d\n\n", N);
    printf("Local  to local:\t%llu cycles\t(%f)\n", tLocalToLocal, ((float) tLocalToLocal) / (float) N);
    printf("Global to global:\t%llu cycles\t(%f)\n", tGlobalToGlobal, ((float) tGlobalToGlobal) / (float) N);
    printf("Shared to shared:\t%llu cycles\t(%f)\n", tSharedToShared, ((float) tSharedToShared) / (float) N);
    printf("Local  to global:\t%llu cycles\t(%f)\n", tLocalToGLobal, ((float) tLocalToGLobal) / (float) N);
    printf("Global to local:\t%llu cycles\t(%f)\n", tglobalToLocal, ((float) tglobalToLocal) / (float) N);
    printf("Shared to global:\t%llu cycles\t(%f)\n", tsharedToGlobal, ((float) tsharedToGlobal) / (float) N);
    printf("Global to shared:\t%llu cycles\t(%f)\n", tglobalToShared, ((float) tglobalToShared) / (float) N);
    printf("Local  to shared:\t%llu cycles\t(%f)\n", tlocalToShared, ((float) tlocalToShared) / (float) N);
    printf("Shared to local:\t%llu cycles\t(%f)\n", tsharedToLocal, ((float) tsharedToLocal) / (float) N);

    return 0;
}