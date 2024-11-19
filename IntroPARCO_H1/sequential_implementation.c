

#include "all_optimization.h"

float* initializeMatrix(int N) {
    
    float* M = (float*)malloc(N * N * sizeof(float));
    if (M == NULL) {
        free(M);
        return NULL;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i * N + j] = (float)rand() / RAND_MAX;
        }
    }
    return M; 
}

bool checkSym(float* M, int N) {  
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (M[i*N+j] != M[j*N+i]) {
                return false;
            }
        }
    }
    return true;
}


float* matTranspose(float* M, int N) {
    float* T = (float*)malloc(N * N * sizeof(float));
    if (T == NULL) {
        free(T);
        return NULL;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T[i*N+j] = M[j*N+i];
        }
    }

    return T;
}
