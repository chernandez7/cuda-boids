
#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <curand.h> // rand function
#include <curand_kernel.h>

#define BlockSize 64

void initCuda(int n);
void flock(int n, int window_width, int window_height, float3 target);
void cudaUpdateVBO(int n, float* vbodptr, float* velptr, float* accptr);

#endif
