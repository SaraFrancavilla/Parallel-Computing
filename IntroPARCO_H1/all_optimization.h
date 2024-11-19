#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#include <immintrin.h>  // For AVX intrinsics

//SEQUENTIAL IMPLEMENTATION
float* initializeMatrix(int N) ;
bool checkSym(float* M, int N) ;
float* matTranspose(float* M, int N) ;


//IMPLICIT PARALLELIZATION
float* matTransposeVectorized(float* M, int N, int BLOCK_SIZE);
float* matTransposePrefetch(float* M, int N, int PREFETCHING_SIZE) ;
float* matTransposeOptimized(float* M, int N, int BLOCK_SIZE);

bool checkSymVectorized(float* M, int N, int BLOCK_SIZE);
bool checkSymPrefetch(float* M, int N, int PREFETCHING_SIZE);
bool checkSymOptimized(float* M, int N, int BLOCK_SIZE);

//OPENMP PARALLELIZATION
float* matTransposeOMP(float* M, int N);
float* matTransposeOMPBlocking(float* M, int N, int BLOCK_SIZE_OMP);
bool checkSymOMP(float* M, int N);
bool checkSymOMPBlocking(float* M, int N, int BLOCK_SIZE_OMP);
float* full_optimized_matTranspose(float* M, int N, int BLOCK_SIZE_OMP);