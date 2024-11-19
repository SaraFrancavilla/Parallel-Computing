#include "all_optimization.h"



float* matTransposeOMP(float* M, int N) {
    float* T = (float*)malloc(N * N * sizeof(float));
    if (T == NULL) {
        return NULL;
    }


    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp simd
        for (int j = 0; j < N; j++) {
            T[i * N + j] = M[j * N + i];
        }
    }

    return T;
}



float* matTransposeOMPBlocking(float* M, int N, int BLOCK_SIZE_OMP) {
    float* T = (float*)malloc(N * N * sizeof(float));
    if (T == NULL) {
        free(T);
        return NULL;
    }

    #pragma omp parallel for 
    for (int i = 0; i < N; i += BLOCK_SIZE_OMP) {
        for (int j = 0; j < N; j += BLOCK_SIZE_OMP) {
            for (int ii = i; ii < i + BLOCK_SIZE_OMP && ii < N; ii++) {
                for (int jj = j; jj < j + BLOCK_SIZE_OMP && jj < N; jj++) {
                    T[ii * N + jj] = M[jj * N + ii];
                }
            }
        }
    }

    return T;
}



bool checkSymOMP(float* M, int N) {
    bool isSymmetric = true;
    bool foundNonSymmetric = false; 

    #pragma omp parallel for shared(foundNonSymmetric)
    for (int i = 0; i < N; i++) {
        // Stop computation
        if (foundNonSymmetric) continue;

        for (int j = i + 1; j < N; j++) {
            if (M[i * N + j] != M[j * N + i]) {
                #pragma omp critical
                {
                    foundNonSymmetric = true;
                }
            }
            if (foundNonSymmetric) break; 
        }
    }

    return !foundNonSymmetric; 
}

bool checkSymOMPBlocking(float* M, int N, int BLOCK_SIZE_OMP) {
    bool isSymmetric = true;
    bool foundNonSymmetric = false;

    // Using blocks
    #pragma omp parallel for shared(foundNonSymmetric)
    for (int i = 0; i < N; i += BLOCK_SIZE_OMP) {
        if (foundNonSymmetric) continue;

        for (int j = i + 1; j < N; j += BLOCK_SIZE_OMP) {
            for (int ii = i; ii < i + BLOCK_SIZE_OMP && ii < N; ii++) {
                for (int jj = j; jj < j + BLOCK_SIZE_OMP && jj < N; jj++) {
                    if (M[ii * N + jj] != M[jj * N + ii]) {
                        #pragma omp critical
                        {
                            foundNonSymmetric = true;
                        }
                        break;
                    }
                }
                if (foundNonSymmetric) break; 
            }
            if (foundNonSymmetric) break; 
        }
    }

    isSymmetric = !foundNonSymmetric;
    return isSymmetric;
}

// Using Blocking, SIMD, and Dynamic Threading
float* full_optimized_matTranspose(float* M, int N, int BLOCK_SIZE_OMP) {

    float* T = (float*)aligned_alloc(32, N * N * sizeof(float)); 
    if (T == NULL) {
        free(T); 
        return NULL; 
    }

    // Adjust thread count based on matrix size
    int num_threads = (N < 512) ? 2 : (N < 2048) ? 6 : 8;  

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        // Dynamic scheduling to improve load balancing
        #pragma omp for collapse(2) schedule(dynamic)  
        for (int i = 0; i < N; i += BLOCK_SIZE_OMP) {
            for (int j = 0; j < N; j += BLOCK_SIZE_OMP) {
                for (int ii = i; ii < i + BLOCK_SIZE_OMP && ii < N; ii++) {
                    // Use AVX vectorization
                    int jj;
                    for (jj = j; jj + 8 <= j + BLOCK_SIZE_OMP && jj + 8 <= N; jj += 8) {
                        __m256 col = _mm256_loadu_ps(&M[ii + jj * N]);
                        _mm256_storeu_ps(&T[jj + ii * N], col);
                    }
                    // Handle any remaining elements
                    for (; jj < j + BLOCK_SIZE_OMP && jj < N; jj++) {
                        T[ii * N + jj] = M[jj * N + ii];
                    }
                }
            }
        }
    }

    return T;  
}

