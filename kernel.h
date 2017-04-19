#ifndef KERNEL_H
#define KERNEL_H

#include <Windows.h>
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

const __device__ float sep_dist = 100;
const __device__ float ali_dist = 400;
const __device__ float coh_dist = 300;

const __device__ float sep_weight = 1.0f;
const __device__ float ali_weight = 1.5f;
const __device__ float coh_weight = 1.0f;

void initCuda(int n);
void flock(int n, int window_width, int window_height, float3 target, bool followMouse, bool naive);
void cudaUpdateVBO(int n, float* vbodptr, float* velptr, float* accptr);

#endif
