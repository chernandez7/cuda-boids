
#include "kernel.h"

dim3 threadsPerBlock(BlockSize);

const float boidMass = 1.0f;
//const float scene_scale = 4e2;
__device__ curandState_t state;
//const __device__ float neighborRadius = 20.0f;
//const __device__ float neighborAngle = 180.0f;
//const __device__ float c_alignment = 2.0f;
//const __device__ float c_separation = 3.0f;
//const __device__ float c_cohesion = 0.005f;
//const __device__ float c_seek = 0.001f;
const __device__ float maxSpeed = 3.0;

float4* dev_pos;
float3* dev_vel;
float3* dev_acc;

__device__
float distanceFormula(float3 myPos, float3 theirPos) {
  float dx = myPos.x - theirPos.x;
  float dy = myPos.y - theirPos.y;
  float dz = myPos.z - theirPos.z;

  float dist = sqrt(dx*dx + dy*dy + dz*dz);
  return dist;
}

__device__
float dotProduct(float3 v1, float3 v2) {
  return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

__device__
void add2Vectors(float3 v1, float3 v2) {
  v1.x += v2.x;
  v1.y += v2.y;
  v1.z += v2.z;
}

__device__
float3 add2VectorsNew(float3 v1, float3 v2) {
  float3 temp = make_float3(v1.x, v1.y, v1.z);
  temp.x += v2.x;
  temp.y += v2.y;
  temp.z += v2.z;
  return temp;
}

__device__
void sub2Vectors(float3 v1, float3 v2) {
  v1.x -= v2.x;
  v1.y -= v2.y;
  v1.z -= v2.z;
}

__device__
float3 sub2VectorsNew(float3 v1, float3 v2) {
  float3 temp = make_float3(v1.x, v1.y, v1.z);
  temp.x -= v2.x;
  temp.y -= v2.y;
  temp.z -= v2.z;
  return temp;
}

__device__
void mulVectorByScalar(float scalar, float3 vector) {
  vector.x *= scalar;
  vector.y *= scalar;
  vector.z *= scalar;
}

__device__
float3 mulVectorByScalarNew(float scalar, float3 vector) {
  // Temp to not overwrite original vector
  float3 temp = make_float3(vector.x, vector.y, vector.z);
  temp.x *= scalar;
  temp.y *= scalar;
  temp.z *= scalar;
  return temp;
}

__device__
void divVectorByScalar(float scalar, float3 vector) {
  vector.x /= scalar;
  vector.y /= scalar;
  vector.z /= scalar;
}

__device__
float3 divVectorByScalarNew(float scalar, float3 vector) {
  float3 temp = make_float3(vector.x, vector.y, vector.z);
  temp.x /= scalar;
  temp.y /= scalar;
  temp.z /= scalar;
  return temp;
}

__device__
void addVectorByScalar(float scalar, float3 vector) {
  vector.x += scalar;
  vector.y += scalar;
  vector.z += scalar;
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
float3 truncate(float3 direction, float maxLength) {

  if (magnitudeOfVector(direction) > maxLength) {
    normalizeVector(direction);
    mulVectorByScalar(maxLength, direction);
    return direction;
  } else {
    return direction;
  }
}

__device__
float3 limit(float max, float3 vector) {
  float size = magnitudeOfVector(vector);
  if (size > max) {
    divVectorByScalar(size, vector);
    return vector;
  } else {
    return vector;
  }
}

__global__
void generateRandomPosArray(int n, float4* arr, float mass) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < n) {
    curand_init(0, blockIdx.x, 0, &state);

    arr[index].x = curand(&state) % 20;
    arr[index].y = curand(&state) % 20;
    arr[index].z = curand(&state) % 20;

    //arr[index].x = index;
    //arr[index].y = index;
    //arr[index].z = index;
    arr[index].w = mass;
  }
}

__global__
void generateRandomVelArray(int n, float3* arr) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < n) {
    curand_init(0, blockIdx.x, 0, &state);

    arr[index].x = curand(&state) % 3;
    arr[index].y = curand(&state) % 3;
    arr[index].z = curand(&state) % 3;
  }
}

__global__
void generateRandomAccArray(int n, float3* arr) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < n) {
    curand_init(0, blockIdx.x, 0, &state);

    arr[index].x = curand(&state) % 3;
    arr[index].y = curand(&state) % 3;
    arr[index].z = curand(&state) % 3;
  }
}

__device__
float3 seek(float3 sum, float3 vel) {
  float3 desired = make_float3(0, 0, 0);
  sub2Vectors(desired, sum);
  normalizeVector(desired);

  mulVectorByScalar(maxSpeed, desired);

  // Steering = desired - vel
  return limit(0.5, sub2VectorsNew(desired, vel));
}

__device__
float3 SeparationRule(int n, float4* pos, float3* vel) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {
    float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);
    float3 myVelocity = make_float3(vel[index].x, vel[index].y, vel[index].z);
    float3 steer = make_float3(0, 0, 0);
    float neighborDistance = 20;
    int numberOfNeighbors = 0;

    // Check against all particles if in range
    for (int i = 0; i < n; i++) {
      float3 theirPos = make_float3(pos[i].x, pos[i].y, pos[i].z);
      float distanceToNeighbor = distanceFormula(myPosition, theirPos);
      // Add the difference of positions to steer force
      if (distanceToNeighbor > 0 && distanceToNeighbor < neighborDistance) {
        float3 deltaPos = sub2VectorsNew(myPosition, theirPos);
        normalizeVector(deltaPos);
        divVectorByScalar(distanceToNeighbor, deltaPos);
        add2Vectors(steer, deltaPos);
        numberOfNeighbors++;
      }
    }
    if (numberOfNeighbors > 0) {
        divVectorByScalar(numberOfNeighbors, steer);
    }
    if (magnitudeOfVector(steer) > 0) {
      normalizeVector(steer);
      mulVectorByScalar(maxSpeed, steer);
      sub2Vectors(steer, myVelocity);
      limit(0.5, steer);
    }
    return steer;
  } else {
    return make_float3(0, 0, 0);
  }
}

__device__
float3 AlignmentRule(int n, float4* pos, float3* vel) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {
    float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);
    float3 myVelocity = make_float3(vel[index].x, vel[index].y, vel[index].z);

    float neighborDistance = 75;
    int numberOfNeighbors = 0;
    float3 sum = make_float3(0, 0, 0);

    // Check against all particles if in range
    for (int i = 0; i < n; i++) {
      float3 theirPos = make_float3(pos[i].x, pos[i].y, pos[i].z);
      float3 theirVel = make_float3(vel[i].x, vel[i].y, vel[i].z);
      // Get distance between you and neighbor
      float distanceToNeighbor = distanceFormula(myPosition, theirPos);
      if (distanceToNeighbor > 0 && distanceToNeighbor < neighborDistance) {
        add2Vectors(sum, theirVel);
        numberOfNeighbors++;
      }
    }
    if (numberOfNeighbors > 0) {
      divVectorByScalar(numberOfNeighbors, sum);
      normalizeVector(sum);
      mulVectorByScalar(maxSpeed, sum);

      return limit(0.5, sub2VectorsNew(sum, myVelocity));
    } else {
      // No neighbors nearby
      return make_float3(0, 0, 0);
    }
  } else {
    return make_float3(0, 0, 0);
  }
}

__device__
float3 CohesionRule(int n, float4* pos, float3* vel) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {

    float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);
    float3 myVelocity = make_float3(vel[index].x, vel[index].y, vel[index].z);

    float neighborDistance = 25;
    int numberOfNeighbors = 0;
    float3 sum = make_float3(0, 0, 0);

    // Check against all particles if in range
    for (int i = 0; i < n; i++) {
      float3 theirPos = make_float3(pos[i].x, pos[i].y, pos[i].z);
      // Get distance between you and neighbor
      float distanceToNeighbor = distanceFormula(myPosition, theirPos);
      // If in range add their pos to sum
      if (distanceToNeighbor > 0 && distanceToNeighbor < neighborDistance) {
        add2Vectors(sum, theirPos);
  			numberOfNeighbors++;
  		}
    }
    if (numberOfNeighbors > 0) {
      divVectorByScalar(numberOfNeighbors, sum);
      return seek(sum, vel[index]);
    } else {
      // No neighbors nearby
      return make_float3(0, 0, 0);
    }
  } else {
    return make_float3(0, 0, 0);
  }
}

__global__
void flocking(int n, float4* pos, float3* vel, float3* acc, float3 target) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {
    // Separation
    float3 separation = SeparationRule(n, pos, vel);
    // Alignment
    float3 alignment = AlignmentRule(n, pos, vel);
    // Cohesion
    float3 cohesion = CohesionRule(n, pos, vel);
    // Apply Arbitrary Weights
    mulVectorByScalar(1.5, separation);
    mulVectorByScalar(1.0, alignment);
    mulVectorByScalar(1.0, cohesion);
    // Apply Forces
    add2Vectors(acc[index], separation);
    add2Vectors(acc[index], alignment);
    add2Vectors(acc[index], cohesion);
  }
}

__device__
void borders(int n, float4* pos, int window_width, int window_height) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {
    // Can't go too far in Z-axis
    if (pos[index].x < 0)    pos[index].x += window_width;
  	if (pos[index].y < 0)    pos[index].y += window_height;
  	if (pos[index].x > window_width) pos[index].x -= window_width;
  	if (pos[index].y > window_height) pos[index].y -= window_height;
  }
}

__global__
void updatePosition(int n, float dt, float4 *pos, float3 *vel, float3 *acc, int window_width, int window_height) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {

    // Slow down won't be as abrupt
    mulVectorByScalar(dt, acc[index]);
    // Update velocity by adding acceleration
    add2Vectors(vel[index], acc[index]);
    // Limit speed
    limit(maxSpeed, vel[index]);

    // Runge- Kutta Method for ODE is a possibility
    // Euler method
    pos[index].x += vel[index].x;
    pos[index].y += vel[index].y;
    pos[index].z += vel[index].z;

    // Reset acc to 0
    mulVectorByScalar(0, acc[index]);
    // Fix positions if particle is off screen
    borders(n, pos, window_width, window_height);
  }
}

__global__
void sendToVBO(int n, float4* pos, float3* vel, float3* acc,
               float* posVBO, float* velVBO, float* accVBO) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < n) {
    posVBO[4 * index + 0] = pos[index].x;
    posVBO[4 * index + 1] = pos[index].y;
    posVBO[4 * index + 2] = pos[index].z;
    posVBO[4 * index + 3] = 1;

    velVBO[3 * index + 0] = vel[index].x;
		velVBO[3 * index + 1] = vel[index].y;
		velVBO[3 * index + 2] = vel[index].z;

    accVBO[3 * index + 0] = acc[index].x;
		accVBO[3 * index + 1] = acc[index].y;
		accVBO[3 * index + 2] = acc[index].z;
  }
}

__host__
void initCuda(int n) {
  //fprintf(stdout, "   Initializing CUDA.\n");
  dim3 fullBlocksPerGrid((int)ceil(float(n)/float(BlockSize)));

  checkCudaErrors( cudaMalloc((void**)&dev_pos, n*sizeof(float4)) );
  checkCudaErrors( cudaMalloc((void**)&dev_vel, n*sizeof(float3)) );
  checkCudaErrors( cudaMalloc((void**)&dev_acc, n*sizeof(float3)) );

  generateRandomPosArray<<<fullBlocksPerGrid, BlockSize>>>(n, dev_pos, boidMass);
  generateRandomVelArray<<<fullBlocksPerGrid, BlockSize>>>(n, dev_vel);
  generateRandomAccArray<<<fullBlocksPerGrid, BlockSize>>>(n, dev_acc);
}

__host__
void flock(int n, int window_width, int window_height, float3 target) {
  dim3 fullBlocksPerGrid((int)ceil(float(n)/float(BlockSize)));

  //fprintf(stdout, "   Updating Acceleration and position.\n");
  flocking<<<fullBlocksPerGrid, BlockSize>>>(n, dev_pos, dev_vel, dev_acc, target);
  updatePosition<<<fullBlocksPerGrid, BlockSize>>>(n, 0.5, dev_pos, dev_vel, dev_acc, window_width, window_height);

}

__host__
void cudaUpdateVBO(int n, float* vbodptr, float* velptr, float* accptr) {
  dim3 fullBlocksPerGrid((int)ceil(float(n)/float(BlockSize)));

  //fprintf(stdout, "   Sending changes to VBO.\n");
  sendToVBO<<<fullBlocksPerGrid, BlockSize>>>(n, dev_pos, dev_vel, dev_acc, vbodptr, velptr, accptr);
}
