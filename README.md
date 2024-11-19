# Parallel-Computing
# IntroPARCO_2024_H1

This project implements various optimizations for sequential and parallel algorithms using OpenMP and AVX2.

## Prerequisites

Before starting, ensure you have the following tools and libraries installed in your environment:

### Required Tools:
- **g++** (preferably version 9.3 or later)
- **WSL** (Windows Subsystem for Linux) if you're working on a Windows system.
- **Make** (optional, if you want to automate the compilation using a Makefile).

### Required Libraries:
- **omp.h** (included with a compiler that supports OpenMP)
- **immintrin.h** (included with the compiler for AVX2 instructions)

To verify the availability of the required libraries, compile and run a simple test program:

```cpp
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>

int main() {
    printf("OpenMP and AVX2 are available!\n");
    return 0;
}
