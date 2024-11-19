#include "all_optimization.h"


// Using AVX intrinsics
float* matTransposeVectorized(float* M, int N, int BLOCK_SIZE) {

    float* T = (float*)aligned_alloc(32, N * N * sizeof(float)); 
    if (T == NULL){
        free(T);
        return NULL;
    }

    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii++) {
                int jj;
                // Handling 8 elements per iteration
                for (jj = j; jj + 8 <= j + BLOCK_SIZE && jj + 8 <= N; jj += 8) {
                    __m256 col = _mm256_loadu_ps(&M[ii + jj * N]);
                    _mm256_storeu_ps(&T[jj + ii * N], col);
                }
                // Remainder loop for elements not processed by AVX
                for (; jj < j + BLOCK_SIZE && jj < N; jj++) {
                    T[ii * N + jj] = M[jj * N + ii];
                }
            }
        }
    }
    return T;
}

float* matTransposePrefetch(float* M, int N, int PREFETCHING_SIZE) {

    float* T = (float*)malloc(N * N * sizeof(float));
    if (T == NULL) {
        free(T);
        return NULL;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __builtin_prefetch(&M[(j + PREFETCHING_SIZE) * N + i], 0, 1); 
            T[i * N + j] = M[j * N + i];
        }
    }

    return T;
}


float* matTransposeOptimized(float* M, int N, int BLOCK_SIZE) { // Uses blocking

    float* T = (float*)malloc(N * N * sizeof(float));
    if (T == NULL) {
        free(T);
        return NULL;
    }

    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii++) {
                for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj++) {
                    T[ii * N + jj] = M[jj * N + ii];
                }
            }
        }
    }
    return T;
}

bool checkSymVectorized(float* M, int N, int BLOCK_SIZE) {
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii++) {
                int jj;
                //Handling 8 elements per iteration
                for (jj = j; jj + 8 <= j + BLOCK_SIZE && jj + 8 <= N; jj += 8) {
                    __m256 row = _mm256_loadu_ps(&M[ii * N + jj]);
                    __m256 col = _mm256_loadu_ps(&M[jj * N + ii]);
                    __m256 cmp = _mm256_cmp_ps(row, col, _CMP_NEQ_OQ);
                    if (!_mm256_testz_ps(cmp, cmp)) {
                        return false;
                    }
                }
                // Remainder loop for elements not processed by AVX
                for (; jj < j + BLOCK_SIZE && jj < N; jj++) {
                    if (M[ii * N + jj] != M[jj * N + ii]) {
                        return false;
                    }
                }
            }
        }
    }
    return true; 
}

bool checkSymPrefetch(float* M, int N, int PREFETCHING_SIZE) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Prefetch the next elements to be accessed
            __builtin_prefetch(&M[i * N + (j + PREFETCHING_SIZE)], 0, 1);
            __builtin_prefetch(&M[(j + PREFETCHING_SIZE) * N + i], 0, 1);
            if (M[i * N + j] != M[j * N + i]) {
                return false;
            }
        }
    }
    return true;
}

bool checkSymOptimized(float* M, int N, int BLOCK_SIZE) {
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii++) {
                for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj++) {
                    if (M[ii * N + jj] != M[jj * N + ii]) {
                        return false; 
                    }
                }
            }
        }
    }
    return true;
}

