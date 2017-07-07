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

__global__ void useLocal(unsigned long long *d_time)
{
    int target = 0;

    int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i * 2 + 1;
    }

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        target += arr[i];
    }
    unsigned long long endTime = clock();

    // Use the local variable so the compiler doesn't optimize it away
    arr[N - 1] = target;
    
    *d_time = (endTime - startTime);
}

__global__ void useGlobal(int *d_v, unsigned long long *d_time)
{
    int target = 0;

    for (int i = 0; i < N; i++) {
        d_v[i] = i * 2 + 1;
    }

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        target += d_v[i];
    }
    unsigned long long endTime = clock();

    // Use the local variable so the compiler doesn't optimize it away
    d_v[N - 1] = target;
    
    *d_time = (endTime - startTime);
}

__global__ void useShared(unsigned long long *d_time)
{
    int target = 0;

    __shared__ int sharedArr[N];
    for (int i = 0; i < N; i++) {
        sharedArr[i] = i;
    }

    __syncthreads();

    unsigned long long startTime = clock();
    for (int i = 0; i < N; i++) {
        target += sharedArr[i];
    }
    unsigned long long endTime = clock();

    // Use the local variable so the compiler doesn't optimize it away
    sharedArr[N - 1] = target;
    
    *d_time = (endTime - startTime);
}

int main()
{
    /**
     * Set up memory on device.
     */
    int *d_useGlobal = NULL;
    if (cudaMalloc((void **) &d_useGlobal, N * sizeof(int)) != cudaSuccess)
        fail("Failed to allocate space for 'd_localToGlobal'");

    unsigned long long tUseLocal;
    unsigned long long tUseGlobal;
    unsigned long long tUseShared;

    unsigned long long *d_tUseLocal = NULL;
    unsigned long long *d_tUseGlobal = NULL;
    unsigned long long *d_tUseShared = NULL;
    if (cudaMalloc((void **) &d_tUseLocal , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tUseLocal'");
    if (cudaMalloc((void **) &d_tUseGlobal , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tUseGlobal'");
    if (cudaMalloc((void **) &d_tUseShared , sizeof(unsigned long long)) != cudaSuccess)
        fail("Failed to allocate space for 'd_tUseShared'");



    /**
     * Execute kernels.
     */
    int nBlocks = 32;
    int nThreads = 128;

    useLocal<<<nBlocks, nThreads>>>(d_tUseLocal);
    useGlobal<<<nBlocks, nThreads>>>(d_useGlobal, d_tUseGlobal);
    useShared<<<nBlocks, nThreads>>>(d_tUseShared);



    /**
     * Copy results back.
     */
    if (cudaMemcpy(&tUseLocal, d_tUseLocal, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tUseLocal");
    if (cudaMemcpy(&tUseGlobal, d_tUseGlobal, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tUseGlobal");
    if (cudaMemcpy(&tUseShared, d_tUseShared, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess)
        fail("Failed to copy to tUseShared");




    /**
     * Print results.
     */
    printf("Benchmark: adding ints to a local variable\n");
    printf("N = %d\n\n", N);
    printf("Using local:\t\t%llu cycles\t(%f)\n", tUseLocal, ((float) tUseLocal) / (float) N);
    printf("Using global:\t\t%llu cycles\t(%f)\n", tUseGlobal, ((float) tUseGlobal) / (float) N);
    printf("Using shared:\t\t%llu cycles\t(%f)\n", tUseShared, ((float) tUseShared) / (float) N);

    return 0;
}