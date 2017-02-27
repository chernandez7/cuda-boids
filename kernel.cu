
#include <stdio.h>
#include <cmath>
#include "kernel.h"

dim3 threadsPerBlock(blockSize);

const float boidMass = 1.0f;
const float scene_scale = 4e2;
const __device__ float neighborRadius = 20.0f;
const __device__ float neighborAngle = 180.0f;
const __device__ float c_alignment = 2.0f;
const __device__ float c_separation = 3.0f;
const __device__ float c_cohesion = 0.005f;
const __device__ float c_seek = 0.001f;

float4* dev_pos;
float3* dev_vel;
float3* dev_acc;

__host__
void initCuda(int n) {
  dim3 fullBlocksPerGrid((int)ceil(float(n)/float(blockSize)));

  checkCudaErrors( cudaMalloc((void**)&dev_pos, n*sizeof(float4)) );
  checkCudaErrors( cudaMalloc((void**)&dev_vel, n*sizeof(float3)) );
  checkCudaErrors( cudaMalloc((void**)&dev_acc, n*sizeof(float3)) );

  randomPositionArray<<<fullBlocksPerGrid, BlockSize>>>(1, n, dev_pos, boidMass);
  randomVelocityArray<<<fullBlocksPerGrid, BlockSize>>>(2, n, dev_vel);
}

__host__
void cudaFlockingUpdateWrapper(int n, float dt, float3 target) {
  dim3 fullBlocksPerGrid((int)ceil(float(n)/float(blockSize)));

  updateAccelaration<<<fullBlocksPerGrid, blockSize>>>(n, dev_pos, dev_vel, dev_acc, target);
  updatePosition<<<fullBlocksPerGrid, blockSize>>>(n, dt, dev_pos, dev_vel, dev_acc);
}

__host__
void cudaUpdateVBO(int n, float* vbodptr, float* velptr) {
  dim3 fullBlocksPerGrid((int)ceil(float(n)/float(blockSize)));

  sendToVBO<<<fullBlocksPerGrid, blockSize>>>(n, dev_pos, dev_vel, vbodptr, velptr, scene_scale);
}

__global__
void generateRandomPosArray(int time, int n, float4* arr, float mass) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < n) {

    arr[index].x = rand() % 3 - 2;
    arr[index].y = rand() % 3 - 2;
    arr[index].z = rand() % 3 - 2;
    arr[index].w = 1.0f;
  }
}

__global__
void generateRandomVelArray(int time, int n, float3* arr) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < n) {

    arr[index].x = rand() % 3 - 2;
    arr[index].y = rand() % 3 - 2;
    arr[index].z = rand() % 3 - 2;
  }
}

__global__
void updateAccelaration(int n, float4* pos, float3* vel,
                        float3* acc, float3 target) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {
    float3 myPosition(pos[index].x, pos[index].y, pos[index].z);
    float3 myVelocity(vel[index].x, vel[index].y, vel[index].z);

    int numberOfNeighbors = 0;
    float3 alignmentNumerator(0.0f, 0.0f, 0.0f);
    float3 alignmentVelocity(0.0f, 0.0f, 0.0f);
    float3 separationVel(0.0f, 0.0f, 0.0f);
    float3 centerOfMass(0.0f, 0.0f, 0.0f);
    float3 desiredVel(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < n; i++) {
      float3 theirPos(pos[i].x, pos[i].y, pos[i].z);
      float distanceToNeighbor = distanceFormula(myPosition, theirPosition);
      if (distanceToNeighbor < neighborRadius &&
        dotProduct(normalizeVector(myVelocity), normalizeVector(sub2Vectors(theirPos, myPosition))) > cos(neighborAngle / 2)) {
        add2Vectors(alignmentNumerator, vel[i]);
        add2Vectors(separationVel, sub2Vectors(myPosition, theirPos) / (distanceToNeighbor * distanceToNeighbor));
        add2Vectors(centerOfMass, theirPos);
        numberOfNeighbors++;
      }
    }
    if (numberOfNeighbors > 0) {
      alignmentVelocity = divVectorByScalar(float(numberOfNeighbors), alignmentNumerator);
      centerOfMass = divVectorByScalar(float(numberOfNeighbors), centerOfMass);
      desiredVel = mulVectorByScalar(c_alignment, alignmentVelocity) +
                   mulVectorByScalar(c_separation, separationVel) +
                   mulVectorByScalar(c_cohesion * sub2Vectors(centerOfMass, myPosition)) +
                   mulVectorByScalar(c_seek * normalizeVector(sub2Vectors(target, myPosition)));
    } else {
      desiredVel = mulVectorByScalar(c_seek, sub2Vectors(target, myPosition));
    }
    if (magnitudeOfVector(myPosition) > 800.0f) {
      desiredVel = normalizeVector(-myPosition);
    }
    acc[index] = truncate(sub2Vectors(desiredVel, myVelocity), 2.0f) / pos[index].w;
  }
}

__global__
void updatePosition(int n, float dt, float4 *pos, float3 *vel, float3 *acc) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {
    vel[index] = mulVectorByScalar(2.0f,
                 normalizeVector(mulVectorByScalar(dt,
                 add2Vectors(vel[index], acc[index])));

    // Runge- Kutta Method for ODE is a possibility

    // Euler method
    pos[index].x += vel[index].x;
    pos[index].y += vel[index].y;
    pos[index].z += vel[index].z;
  }
}

__global__
void sendToVBO(int n, float4* pos, float3* vel,
               float* posVBO, float* velVBO, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale_w = 2.0f / s_scale;
  float c_scale_h = 2.0f / s_scale;
	float c_scale_s = 2.0f / s_scale;

  if (index < n) {
    posVBO[4 * index + 0] = pos[index].x * c_scale_w;
    posVBO[4 * index + 1] = pos[index].y * c_scale_h;
    posVBO[4 * index + 2] = pos[index].z * c_scale_s;
    posVBO[4 * index + 3] = 1;

    velVBO[3 * index + 0] = vel[index].x * c_scale_w;
		velVBO[3 * index + 1] = vel[index].y * c_scale_h;
		velVBO[3 * index + 2] = vel[index].z * c_scale_s;
  }
}

__device__
float3 truncate(float3 direction, float maxLength) {
  if (magnitudeOfVector(direction) > maxLength) {
    return normalizeVector(direction) * maxLength;
  } else {
    return direction;
  }
}

__device__
float distanceFormula(float3 myPos, float3 theirPos) {
  float dx = myPos.x - theirPos.x;
  float dy = myPos.y - theirPos.y;
  float dz = myPos.z - theirPos.z;

  float dist = sqrt(dx*dx + dy*dy + dz*dz)
  return dist;
}

__device__
float magnitudeOfVector(float3 vector) {
  return sqrt(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);
}

__device__
void normalizeVector(float3 vector) {
  float magnitude = magnitudeOfVector(vector);
  if (magnitude > 0) {
    vector.x /= magnitude;
    vector.y /= magnitude;
    vector.z /= magnitude;
  }
}

__device__
float dotProduct(float3 v1, float3 v2) {
  return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

__device__
void add2Vectors(float3 v1, float3 v2) {
  float v1.x += v2.x;
  float v1.y += v2.y;
  float v1.z += v2.z;
}

__device__
void sub2Vectors(float3 v1, float3 v2) {
  float v1.x -= v2.x;
  float v1.y -= v2.y;
  float v1.z -= v2.z;
}

__device__
void mulVectorByScalar(float scalar, float3 vector) {
  vector.x *= scalar;
  vector.y *= scalar;
  vector.z *= scalar;
}

__device__
void divVectorByScalar(float scalar, float3 vector) {
  vector.x /= scalar;
  vector.y /= scalar;
  vector.z /= scalar;
}

__device__
void addVectorByScalar(float scalar, float3 vector) {
  vector.x += scalar;
  vector.y += scalar;
  vector.z += scalar;
}
