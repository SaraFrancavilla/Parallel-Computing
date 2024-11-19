# Parallel-Computing

# Reproducing IntroPARCO_2024_H1

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
```
Compile the test program using:

```bash
g++ -fopenmp -mavx2 test.c -o test
```
## Compilation Instructions

To compile the project, use the following command:

```bash
g++ -O3 -mavx2 -march=native -fopenmp -ffast-math -funroll-loops introPARCO_2024_H1.c all_optimization.h sequential_implementation.c implicit_parallelization.c explicit_parallelization.c -o introPARCO_2024
```

### Explanation of the Flags:

- ```-O3 ```: Enables high-level optimizations.

- ```-mavx2```: Enables AVX2 instructions.
- ```-march=native```: Utilizes features specific to your CPU.
- ```-fopenmp```: Enables OpenMP support.
- ```-ffast-math```: Improves the speed of math operations (may slightly reduce precision).
- ```-funroll-loops```: Expands loops for performance improvement.

This will create an executable named ```introPARCO_2024```.

## Running the Program

To run the program, use the following command:

Running the Program
To run the program, use the following command:

```bash
./introPARCO_2024
```

## Code Structure
The project consists of the following files:

- **introPARCO_2024_H1.c**: The main entry point of the program.
- **all_optimization.h**: Header file containing common definitions and macros.
- **sequential_implementation.c**: Implementation of the sequential version of the algorithms.
- **implicit_parallelization.c**: Parallel implementation using an implicit approach (e.g., OpenMP).
- **explicit_parallelization.c**: Parallel implementation using explicit AVX2 intrinsics.

## Common Issues
1. **Error: ```#include <omp.h> ```not found**

- Ensure you are using a compiler that supports OpenMP (e.g., ```g++```).
-On Ubuntu, you can install it with:

```bash
sudo apt update
sudo apt install g++
```
2. **Error:``` #include <immintrin.h> ```not found**
- Ensure your compiler supports AVX2 instructions. Check your CPU support using:
```bash
lscpu | grep avx2
```
If ```AVX2``` is not listed, your hardware does not support these instructions.
3. **Slow performance**
- Ensure you are using the ```-march=native``` flag and that your system supports AVX2.




