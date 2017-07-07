#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <semaphore.h>

#include "../matrix.c"
#include "../matrix.h"

void fail(char *message)
{
    printf(message);
    exit(EXIT_FAILURE);
}

sem_t *gettingWork;

int nextID = 0;

/**
 * Leave logic of when work is over up to another function.
 */
int getIndex()
{
    sem_wait(gettingWork);

    int toReturn = nextID;
    nextID++;

    sem_post(gettingWork);

    return toReturn;
}

typedef struct {
    int nThreads;
    Matrix *result;
    Matrix *a;
    Matrix *b;
} MultiplyArgs;

void *doMultiply(void *arguments)
{
    int idx = getIndex(); // Thread-safe

    // To make the loop cleaner
    MultiplyArgs *args = (MultiplyArgs *) arguments;
    Matrix *result = args->result;
    Matrix *a = args->a;
    Matrix *b = args->b;

    int nElements = result->nRows * result->nCols;

    while (idx < nElements) {
        int r = idx / result->nCols; // Integer division on purpose
        int c = idx % result->nCols;

        int *v1 = a->values + (r * a->nCols);
        int *v2 = b->values + (c * b->nRows);
        setValue(result, r, c, dotProduct(v1, v2, a->nCols));

        idx += args->nThreads;
    }

    return NULL;
}

/**
 * Assumes a is row-major and b is column-major.
 */
Matrix *multiply(Matrix *a, Matrix *b, bool resultRowMajor)
{
    Matrix *result = generateMatrix(a->nRows, b->nCols, resultRowMajor);

    // Set up and start the threads
    int nThreads = 16;
    pthread_t threads[nThreads];

    // Set up the arguments (these won't change)
    MultiplyArgs *args = (MultiplyArgs *) malloc(sizeof(MultiplyArgs));
    args->result = result;
    args->a = a;
    args->b = b;
    args->nThreads = nThreads;
    
    // Start the threads
    for (int i = 0; i < nThreads; i++) {
        if (pthread_create(&(threads[i]), NULL, doMultiply, args) != 0)
            fail("Couldn't create a multiply thread");
    }

    // Wait for them to finish
    for (int i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
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
    
    // Set up the 'get work' semaphore
    gettingWork = malloc(sizeof(sem_t));
    if (sem_init(gettingWork, 0, 1) != 0)
        fail("Couldn't create gettingWork semaphore");

    // Multiply in a threaded way
    Matrix *ab = multiply(a, b, true);

    // Print all matrices
    // printMatrix(a);
    // printf("\n");
    // printMatrix(b);
    // printf("\n");
    // printMatrix(ab);
    
    fflush(stdout);
    fflush(stderr);

    // Clean up
    destroyMatrix(a);
    destroyMatrix(b);
    destroyMatrix(ab);

    sem_close(gettingWork);

    return EXIT_SUCCESS;
}
