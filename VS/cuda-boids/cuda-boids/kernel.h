#ifndef KERNEL_H
#define KERNEL_H

#ifdef _WIN32
#include <Windows.h>
#endif
#include <stdio.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <curand.h> // rand function
#include <curand_kernel.h>

#define BlockSize 256

const float boidMass = 2.0f;
const float scene_scale = 4e2;

const float __device__ maxVelocity = 2.0f; // acc over time
const float __device__ maxSteer = 0.1f; // turn radius

__device__ float dev_sep_dist;
__device__ float dev_ali_dist;
__device__ float dev_coh_dist;

__device__ float dev_sep_weight;
__device__ float dev_ali_weight;
__device__ float dev_coh_weight;

void initCuda(int n);
void flock(int n, int window_width, int window_height, float3 target, bool followMouse, bool naive, float sep_dist, float sep_weight, float ali_dist, float ali_weight, float coh_dist, float coh_weight);
void cudaUpdateVBO(int n, float* vbodptr, float* velptr, float* accptr);

#endif