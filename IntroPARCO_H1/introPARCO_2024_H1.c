#include "all_optimization.h"

int main() {

    float theoretical_peak_performance = 8*8*2.688; // 8 cores, 8 threads per core, 2.688 GHz
    int theoretical_peak_memory_bandwidth = 3200*16*2; // 3200 MHz, 16 bytes per cycle, 2 channels

    float* M_8 = initializeMatrix(8);
    float* M_16 = initializeMatrix(16);
    float* M_32 = initializeMatrix(32);
    float* M_64 = initializeMatrix(64);
    float* M_128 = initializeMatrix(128);
    float* M_256 = initializeMatrix(256);
    float* M_512 = initializeMatrix(512);
    float* M_1024 = initializeMatrix(1024);
    float* M_2048 = initializeMatrix(2048);
    float* M_4096 = initializeMatrix(4096);

    struct timespec start, end;
    
    omp_set_num_threads(6); // Setting number of threads for OpenMP

    printf("\nEVALUATING THE PERFORMANCES AND MEASURING TIME TAKEN FOR MATRIX TRANSPOSITION\n\n");

    printf("Theoretical peak performance: %.9f FLOP/s\n", theoretical_peak_performance);
    printf("Theoretical peak memory bandwidth: %d MB/s\n\n", theoretical_peak_memory_bandwidth);
    printf("B = Bandwidth MB\n\n");
    //Basic formula used to compute memory bandwidth for all matTransposed : 2*N*N*sizeof(float) / time_taken*1000000


    printf("SEQUENTIAL IMPLEMENTATION\n");


    //Measuring checkSym performances with different dimensions matrices

    clock_gettime(CLOCK_MONOTONIC, &start);
    bool isSymmetric_16 = checkSym(M_16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    bool isSymmetric_64 = checkSym(M_64, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    bool isSymmetric_512 = checkSym(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    bool isSymmetric_4096 = checkSym(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;


    printf("                  N = 16   |   N = 64   |   N = 512   |  N = 4096\n");
    printf("checkSym     : %.9f | %.9f| %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);


    //Measuring matTranspose performances with different dimensions matrices
    clock_gettime(CLOCK_MONOTONIC, &start);
    float* T_16 = matTranspose(M_16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    float* T_64 = matTranspose(M_64, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    float* T_512 = matTranspose(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    float* T_4096 = matTranspose(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    float B_16 = (2*16*16*sizeof(float)) / (time_taken_16*1000000);
    float B_64 = (2*64*64*sizeof(float)) / (time_taken_64*1000000);
    float B_512 = (2*512*512*sizeof(float)) / (time_taken_512*1000000);
    float B_4096 = (2*4096*4096*sizeof(float)) /(time_taken_4096*1000000);
    
    printf("matTranspose : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);
    printf("   bandwidth : %11.4f | %11.4f | %11.4f | %11.4f\n", B_16, B_64, B_512, B_4096);

    printf("============================\n");

    printf("IMPLICIT IMPLEMENTATION\n");

    //Measuring checkSymVectorized performances with different dimensions matrices

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSymVectorized(M_16, 16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSymVectorized(M_64, 64, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymVectorized(M_512, 512, 128);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymVectorized(M_4096, 4096, 256);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("                        N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("checkSymVectorized : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);


    //Measuring checkSymPrefetch performances with different dimensions matrices

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSymPrefetch(M_16, 16, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSymPrefetch(M_64, 64, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymPrefetch(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymPrefetch(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("checkSymPrefetch   : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);


    //Measuring checkSymOptimized performances with different dimensions matrices

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSymOptimized(M_16, 16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSymOptimized(M_64, 64, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOptimized(M_512, 512, 128);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOptimized(M_4096, 4096, 256);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("checkSymOptimized  : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);

    printf("----------------------------\n");

    //Measuring matTransposeVectorized performances with different dimensions matrices
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = matTransposeVectorized(M_16, 16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = matTransposeVectorized(M_64, 64, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeVectorized(M_512, 512, 128);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeVectorized(M_4096, 4096, 256);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    B_16 = (2*16*16*sizeof(float)) / (time_taken_16*1000000);
    B_64 = (2*64*64*sizeof(float)) / (time_taken_64*1000000);
    B_512 = (2*512*512*sizeof(float)) / (time_taken_512*1000000);
    B_4096 = (2*4096*4096*sizeof(float)) /(time_taken_4096*1000000);
    
    printf("                            N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("matTransposeVectorized : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);
    printf("             bandwidth : %11.4f | %11.4f | %11.4f | %11.4f\n", B_16, B_64, B_512, B_4096);


    //Measuring matTransposePrefetch performances with different dimensions matrices
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = matTransposePrefetch(M_16, 16, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = matTransposePrefetch(M_64, 64, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposePrefetch(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposePrefetch(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    
    B_16 = (2*16*16*sizeof(float)) / (time_taken_16*1000000);
    B_64 = (2*64*64*sizeof(float)) / (time_taken_64*1000000);
    B_512 = (2*512*512*sizeof(float)) / (time_taken_512*1000000);
    B_4096 = (2*4096*4096*sizeof(float)) /(time_taken_4096*1000000);
    
    printf("matTransposePrefetch   : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);
    printf("             bandwidth : %11.4f | %11.4f | %11.4f | %11.4f\n", B_16, B_64, B_512, B_4096);



    //Measuring matTransposeOptimized performances with different dimensions matrices
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = matTransposeOptimized(M_16, 16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = matTransposeOptimized(M_64, 64, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOptimized(M_512, 512, 128);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOptimized(M_4096, 4096, 256);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    B_16 = (2*16*16*sizeof(float)) / (time_taken_16*1000000);
    B_64 = (2*64*64*sizeof(float)) / (time_taken_64*1000000);
    B_512 = (2*512*512*sizeof(float)) / (time_taken_512*1000000);
    B_4096 = (2*4096*4096*sizeof(float)) /(time_taken_4096*1000000);
    
    printf("matTransposeOptimized  : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);
    printf("             bandwidth : %11.4f | %11.4f | %11.4f | %11.4f\n", B_16, B_64, B_512, B_4096);

    printf("============================\n");

    printf("OPENMP IMPLEMENTATION\n");

    //Measuring checkSymOMP performances with different dimensions matrices

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSymOMP(M_16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSymOMP(M_64, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("                         N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("checkSymOMP         : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);

     //Measuring checkSymOMPBlocking performances with different dimensions matrices

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSymOMPBlocking(M_16, 16, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSymOMPBlocking(M_64, 64, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("checkSymOMPBlocking : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);

    printf("----------------------------\n");

    //Measuring matTransposeOMP performances with different dimensions matrices
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = matTransposeOMP(M_16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = matTransposeOMP(M_64, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    B_16 = (2*16*16*sizeof(float)) / (time_taken_16*1000000);
    B_64 = (2*64*64*sizeof(float)) / (time_taken_64*1000000);
    B_512 = (2*512*512*sizeof(float)) / (time_taken_512*1000000);
    B_4096 = (2*4096*4096*sizeof(float)) /(time_taken_4096*1000000);
    

    printf("                               N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("matTransposeOMP           : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);
    printf("                bandwidth : %11.4f | %11.4f | %11.4f | %11.4f\n", B_16, B_64, B_512, B_4096);


    //Measuring matTransposeOMPBlocking performances with different dimensions matrices
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = matTransposeOMPBlocking(M_16, 16, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = matTransposeOMPBlocking(M_64, 64, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    B_16 = (2*16*16*sizeof(float)) / (time_taken_16*1000000);
    B_64 = (2*64*64*sizeof(float)) / (time_taken_64*1000000);
    B_512 = (2*512*512*sizeof(float)) / (time_taken_512*1000000);
    B_4096 = (2*4096*4096*sizeof(float)) /(time_taken_4096*1000000);
    
    printf("matTransposeOMPBlocking   : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);
    printf("                bandwidth : %11.4f | %11.4f | %11.4f | %11.4f\n", B_16, B_64, B_512, B_4096);

    printf("==========================================================================================\n\n");

    printf("COMPUTING SPEEDUP AND EFFICIENCY GAINS FOR THE OPENMP IMPLEMENTATION\n\n");
    printf("S = Speedup    E = Efficiency\n\n");

    //Computing time needed for sequential implementation
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSym(M_16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_16_seq = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSym(M_64, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_64_seq = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSym(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_512_seq = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSym(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_4096_seq = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    

    omp_set_num_threads(2); // Working with 2 threads

    printf("2 THREADS\n\n");

    //Measuring checkSymOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSymOMP(M_16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSymOMP(M_64, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    double S_16 = time_taken_16_seq/time_taken_16;
    double S_64 = time_taken_64_seq/time_taken_64;
    double S_512 = time_taken_512_seq/time_taken_512;
    double S_4096 = time_taken_4096_seq/time_taken_4096;
    double E_16 = S_16/2;
    double E_64 = S_64/2;
    double E_512 = S_512/2;
    double E_4096 = S_4096/2;

    printf("                           N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("checkSymOMP S         : %.9f | %.9f | %.9f | %.9f\n", S_16, S_64, S_512, S_4096);
    printf("checkSymOMP E         : %.9f | %.9f | %.9f | %.9f\n", E_16, E_64, E_512, E_4096);

    //Measuring checkSymOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSymOMPBlocking(M_16, 16, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSymOMPBlocking(M_64, 64, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    
    S_16 = time_taken_16_seq/time_taken_16;
    S_64 = time_taken_64_seq/time_taken_64;
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_16 = S_16/2;
    E_64 = S_64/2;
    E_512 = S_512/2;
    E_4096 = S_4096/2;

    printf("checkSymOMPBlocking S : %.9f | %.9f | %.9f | %.9f\n", S_16, S_64, S_512, S_4096);
    printf("checkSymOMPBlocking E : %.9f | %.9f | %.9f | %.9f\n", E_16, E_64, E_512, E_4096);

    printf("----------------------------\n");

    //Measuring matTransposeOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = matTransposeOMP(M_16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = matTransposeOMP(M_64, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    S_16 = time_taken_16_seq/time_taken_16;
    S_64 = time_taken_64_seq/time_taken_64;
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_16 = S_16/2;
    E_64 = S_64/2;
    E_512 = S_512/2;
    E_4096 = S_4096/2;
    
    printf("                               N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("matTransposeOMP S         : %.9f | %.9f | %.9f | %.9f\n", S_16, S_64, S_512, S_4096);
    printf("matTransposeOMP E         : %.9f | %.9f | %.9f | %.9f\n", E_16, E_64, E_512, E_4096);

    //Measuring matTransposeOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = matTransposeOMPBlocking(M_16, 16, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = matTransposeOMPBlocking(M_64, 64, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_16 = time_taken_16_seq/time_taken_16;
    S_64 = time_taken_64_seq/time_taken_64;
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_16 = S_16/2;
    E_64 = S_64/2;
    E_512 = S_512/2;
    E_4096 = S_4096/2;
    
    printf("matTransposeOMPBlocking S : %.9f | %.9f | %.9f | %.9f\n", S_16, S_64, S_512, S_4096);
    printf("matTransposeOMPBlocking E : %.9f | %.9f | %.9f | %.9f\n", E_16, E_64, E_512, E_4096);

    printf("============================\n\n");

    omp_set_num_threads(4); // Working with 4 threads

    printf("4 THREADS\n\n");

    //Measuring checkSymOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSymOMP(M_16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSymOMP(M_64, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_16 = time_taken_16_seq/time_taken_16;
    S_64 = time_taken_64_seq/time_taken_64;
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_16 = S_16/4;
    E_64 = S_64/4;
    E_512 = S_512/4;
    E_4096 = S_4096/4;

    printf("                           N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("checkSymOMP S         : %.9f | %.9f | %.9f | %.9f\n", S_16, S_64, S_512, S_4096);
    printf("checkSymOMP E         : %.9f | %.9f | %.9f | %.9f\n", E_16, E_64, E_512, E_4096);

    //Measuring checkSymOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_16 = checkSymOMPBlocking(M_16, 16, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_64 = checkSymOMPBlocking(M_64, 64, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    
    S_16 = time_taken_16_seq/time_taken_16;
    S_64 = time_taken_64_seq/time_taken_64;
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_16 = S_16/4;
    E_64 = S_64/4;
    E_512 = S_512/4;
    E_4096 = S_4096/4;

    printf("checkSymOMPBlocking S : %.9f | %.9f | %.9f | %.9f\n", S_16, S_64, S_512, S_4096);
    printf("checkSymOMPBlocking E : %.9f | %.9f | %.9f | %.9f\n", E_16, E_64, E_512, E_4096);

    printf("----------------------------\n");

    //Measuring matTransposeOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = matTransposeOMP(M_16, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = matTransposeOMP(M_64, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    S_16 = time_taken_16_seq/time_taken_16;
    S_64 = time_taken_64_seq/time_taken_64;
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_16 = S_16/4;
    E_64 = S_64/4;
    E_512 = S_512/4;
    E_4096 = S_4096/4;
    
    printf("                               N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("matTransposeOMP S         : %.9f | %.9f | %.9f | %.9f\n", S_16, S_64, S_512, S_4096);
    printf("matTransposeOMP E         : %.9f | %.9f | %.9f | %.9f\n", E_16, E_64, E_512, E_4096);

    //Measuring matTransposeOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = matTransposeOMPBlocking(M_16, 16, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = matTransposeOMPBlocking(M_64, 64, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_16 = time_taken_16_seq/time_taken_16;
    S_64 = time_taken_64_seq/time_taken_64;
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_16 = S_16/4;
    E_64 = S_64/4;
    E_512 = S_512/4;
    E_4096 = S_4096/4;
    
    printf("matTransposeOMPBlocking S : %.9f | %.9f | %.9f | %.9f\n", S_16, S_64, S_512, S_4096);
    printf("matTransposeOMPBlocking E : %.9f | %.9f | %.9f | %.9f\n", E_16, E_64, E_512, E_4096);

    printf("============================\n\n");

    omp_set_num_threads(6); // Working with 6 threads

    printf("6 THREADS\n\n");

    //Measuring checkSymOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/6;
    E_4096 = S_4096/6;

    printf("                          N = 512   |  N = 4096\n");
    printf("checkSymOMP S         : %.9f | %.9f\n", S_512, S_4096);
    printf("checkSymOMP E         : %.9f | %.9f\n", E_512, E_4096);

    //Measuring checkSymOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
   
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/6;
    E_4096 = S_4096/6;

    printf("checkSymOMPBlocking S : %.9f | %.9f\n", S_512, S_4096);
    printf("checkSymOMPBlocking E : %.9f | %.9f\n", E_512, E_4096);

    printf("----------------------------\n");

    //Measuring matTransposeOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/6;
    E_4096 = S_4096/6;
    
    printf("                              N = 512   |  N = 4096\n");
    printf("matTransposeOMP S         : %.9f | %.9f\n", S_512, S_4096);
    printf("matTransposeOMP E         : %.9f | %.9f\n", E_512, E_4096);

    //Measuring matTransposeOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/6;
    E_4096 = S_4096/6;
    
    printf("matTransposeOMPBlocking S : %.9f | %.9f\n", S_512, S_4096);
    printf("matTransposeOMPBlocking E : %.9f | %.9f\n", E_512, E_4096);

    printf("============================\n\n");

    omp_set_num_threads(8); // Working with 8 threads

    printf("8 THREADS\n\n");

    //Measuring checkSymOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/8;
    E_4096 = S_4096/8;

    printf("                          N = 512   |  N = 4096\n");
    printf("checkSymOMP S         : %.9f | %.9f\n", S_512, S_4096);
    printf("checkSymOMP E         : %.9f | %.9f\n", E_512, E_4096);

    //Measuring checkSymOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
   
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/8;
    E_4096 = S_4096/8;

    printf("checkSymOMPBlocking S : %.9f | %.9f\n", S_512, S_4096);
    printf("checkSymOMPBlocking E : %.9f | %.9f\n", E_512, E_4096);

    printf("----------------------------\n");

    //Measuring matTransposeOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/8;
    E_4096 = S_4096/8;
    
    printf("                              N = 512   |  N = 4096\n");
    printf("matTransposeOMP S         : %.9f | %.9f\n", S_512, S_4096);
    printf("matTransposeOMP E         : %.9f | %.9f\n", E_512, E_4096);

    //Measuring matTransposeOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/8;
    E_4096 = S_4096/8;
    
    printf("matTransposeOMPBlocking S : %.9f | %.9f\n", S_512, S_4096);
    printf("matTransposeOMPBlocking E : %.9f | %.9f\n", E_512, E_4096);

    printf("============================\n\n");

    omp_set_num_threads(12); // Working with 12 threads

    printf("12 THREADS\n\n");

    //Measuring checkSymOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/12;
    E_4096 = S_4096/12;

    printf("                          N = 512   |  N = 4096\n");
    printf("checkSymOMP S         : %.9f | %.9f\n", S_512, S_4096);
    printf("checkSymOMP E         : %.9f | %.9f\n", E_512, E_4096);

    //Measuring checkSymOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
   
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/12;
    E_4096 = S_4096/12;

    printf("checkSymOMPBlocking S : %.9f | %.9f\n", S_512, S_4096);
    printf("checkSymOMPBlocking E : %.9f | %.9f\n", E_512, E_4096);

    printf("----------------------------\n");

    //Measuring matTransposeOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/12;
    E_4096 = S_4096/12;
    
    printf("                              N = 512   |  N = 4096\n");
    printf("matTransposeOMP S         : %.9f | %.9f\n", S_512, S_4096);
    printf("matTransposeOMP E         : %.9f | %.9f\n", E_512, E_4096);

    //Measuring matTransposeOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/12;
    E_4096 = S_4096/12;
    
    printf("matTransposeOMPBlocking S : %.9f | %.9f\n", S_512, S_4096);
    printf("matTransposeOMPBlocking E : %.9f | %.9f\n", E_512, E_4096);

    printf("============================\n\n");

    omp_set_num_threads(16); // Working with 16 threads

    printf("16 THREADS\n\n");

    //Measuring checkSymOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/16;
    E_4096 = S_4096/16;

    printf("                          N = 512   |  N = 4096\n");
    printf("checkSymOMP S         : %.9f | %.9f\n", S_512, S_4096);
    printf("checkSymOMP E         : %.9f | %.9f\n", E_512, E_4096);

    //Measuring checkSymOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_512 = checkSymOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    isSymmetric_4096 = checkSymOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
   
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/16;
    E_4096 = S_4096/16;

    printf("checkSymOMPBlocking S : %.9f | %.9f\n", S_512, S_4096);
    printf("checkSymOMPBlocking E : %.9f | %.9f\n", E_512, E_4096);

    printf("----------------------------\n");

    //Measuring matTransposeOMP speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMP(M_512, 512);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMP(M_4096, 4096);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/16;
    E_4096 = S_4096/16;
    
    printf("                              N = 512   |  N = 4096\n");
    printf("matTransposeOMP S         : %.9f | %.9f\n", S_512, S_4096);
    printf("matTransposeOMP E         : %.9f | %.9f\n", E_512, E_4096);

    //Measuring matTransposeOMPBlocking speedup and efficiency
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = matTransposeOMPBlocking(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = matTransposeOMPBlocking(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_512 = S_512/16;
    E_4096 = S_4096/16;
    
    printf("matTransposeOMPBlocking S : %.9f | %.9f\n", S_512, S_4096);
    printf("matTransposeOMPBlocking E : %.9f | %.9f\n\n", E_512, E_4096);


    printf("===============================================================================================\n\n");

    //Testing full_full_optimized_matTranspose
    
    printf("FULL OPTIMIZED IMPLEMENTATION\n\n");

    printf("TIME:\n");

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_16 = full_optimized_matTranspose(M_16, 16, 8);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_16 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_64 = full_optimized_matTranspose(M_64, 64, 16);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_64 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    T_512 = full_optimized_matTranspose(M_512, 512, 32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_512 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    T_4096 = full_optimized_matTranspose(M_4096, 4096, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken_4096 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    S_16 = time_taken_16_seq/time_taken_16;
    S_64 = time_taken_64_seq/time_taken_64;
    S_512 = time_taken_512_seq/time_taken_512;
    S_4096 = time_taken_4096_seq/time_taken_4096;
    E_16 = S_16/2;
    E_64 = S_64/2;
    E_512 = S_512/6;
    E_4096 = S_4096/8;

    B_16 = (2*16*16*sizeof(float)) / (time_taken_16*1000000);
    B_64 = (2*64*64*sizeof(float)) / (time_taken_64*1000000);
    B_512 = (2*512*512*sizeof(float)) / (time_taken_512*1000000);
    B_4096 = (2*4096*4096*sizeof(float)) /(time_taken_4096*1000000);


    printf("                                 N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("full_optimized_matTranspose : %.9f | %.9f | %.9f | %.9f\n", time_taken_16, time_taken_64, time_taken_512, time_taken_4096);
    printf("                  bandwidth : %11.4f | %11.4f | %11.4f | %11.4f\n", B_16, B_64, B_512, B_4096);

    printf("-------------------------------------------------------------------------\n\n");

    printf("SPEEDUP AND EFFICIENCY:\n");

    printf("                                   N = 16   |    N = 64   |   N = 512   |  N = 4096\n");
    printf("full_optimized_matTranspose S : %.9f | %.9f | %.9f | %.9f\n", S_16, S_64, S_512, S_4096);
    printf("full_optimized_matTranspose E : %.9f | %.9f | %.9f | %.9f\n\n\n", E_16, E_64, E_512, E_4096);


    free(M_8);
    free(M_16);
    free(M_32);
    free(M_64);
    free(M_128);
    free(M_256);
    free(M_512);
    free(M_1024);
    free(M_2048);
    free(M_4096);
    free(T_16);
    free(T_64);
    free(T_512);
    free(T_4096);


    return 0;
}