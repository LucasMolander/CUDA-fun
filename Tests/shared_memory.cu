#include <stdio.h>

#include <cuda_runtime.h>

#define TILE_SIZE (32)

void fail(const char *message)
{
    printf(message);
    fflush(stdout);
    exit(EXIT_FAILURE);
}

__global__ void useSharedMemory()
{
	__shared__ int arrOne[TILE_SIZE][TILE_SIZE];
	__shared__ int arrTwo[TILE_SIZE][TILE_SIZE];

	// Get rid of compiler warnings, and try to avoid getting optimized away
	arrTwo[1][0] = clock() % 1000;
	arrOne[1][0] = clock() % 1000;

	arrOne[0][1] = arrTwo[1][0];
	arrTwo[0][1] = arrOne[1][0];
}

int main()
{
	// Why isn't it breaking?????
	
	int sMemBytes = 114688; // 112 KB
	int nBlocks = 128;
	int nThreadsPerBlock = TILE_SIZE * TILE_SIZE;
	int sBytesPerBlock = sMemBytes / nBlocks;

	printf("Shared memory bytes total: %d\n", sMemBytes);
	printf("Number of blocks: %d\n", nBlocks);
	printf("Threads per block: %d\n", nThreadsPerBlock);
	printf("Shared bytes per block: %d\n", sBytesPerBlock);
	printf("\n");
	printf("Tile size: %d\n", TILE_SIZE);
	printf("One array bytes: %d\n", TILE_SIZE * TILE_SIZE * sizeof(int));
	printf("Two arrays bytes: %d\n", 2 * TILE_SIZE * TILE_SIZE * sizeof(int));

	printf("\nuseSharedMemory<<<%d, %d>>>();\n", nBlocks, nThreadsPerBlock);
	useSharedMemory<<<nBlocks, nThreadsPerBlock>>>();

	if (cudaGetLastError() != cudaSuccess)
        fail("Failure in CUDA kernel execution\n");

    printf("\nRan kernel successfully!\n");

	return 0;
}
