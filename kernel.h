
#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <cmath>

#define blockSize 128

void initCuda(int n);
void cudaFlockingUpdateWrapper(int n, float dt, float3 target);
void cudaUpdateVBO(int n, float *vbodptr, float *velptr);

#endif
