
#include "kernel.h"

dim3 threadsPerBlock(BlockSize);

const float boidMass = 1.0f;
const float scene_scale = 4e2;
__device__ curandState_t state;
const __device__ float neighborRadius = 20.0f;
const __device__ float neighborAngle = 180.0f;
const __device__ float c_alignment = 2.0f;
const __device__ float c_separation = 3.0f;
const __device__ float c_cohesion = 0.005f;
const __device__ float c_seek = 0.001f;

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

__global__
void generateRandomPosArray(int n, float4* arr, float mass) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < n) {
    curand_init(0, blockIdx.x, 0, &state);

    arr[index].x = curand(&state) % 20;
    arr[index].y = curand(&state) % 20;
    arr[index].z = curand(&state) % 20;

    arr[index].x = index;
    arr[index].y = index;
    arr[index].z = index;
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
void updateAccelaration(int n, float4* pos, float3* vel,
                        float3* acc, float3 target) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {

    float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);
    float3 myVelocity = make_float3(vel[index].x, vel[index].y, vel[index].z);

    int numberOfNeighbors = 0;
    float3 alignmentNumerator = make_float3(0.0f, 0.0f, 0.0f);
    float3 alignmentVelocity = make_float3(0.0f, 0.0f, 0.0f);
    float3 separationVel = make_float3(0.0f, 0.0f, 0.0f);
    float3 centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    float3 desiredVel = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < n; i++) {
      float3 theirPos = make_float3(pos[i].x, pos[i].y, pos[i].z);
      float distanceToNeighbor = distanceFormula(myPosition, theirPos);

      if (distanceToNeighbor > 0 && distanceToNeighbor < neighborRadius) {
	      normalizeVector(myVelocity);
	      float3 deltaPos = sub2VectorsNew(theirPos, myPosition);
        normalizeVector(deltaPos);
      	if (dotProduct(myVelocity, deltaPos) > cos(neighborAngle / 2)) {
          add2Vectors(alignmentNumerator, vel[i]);

      	  float3 reverseDelta = sub2VectorsNew(myPosition, theirPos);
      	  divVectorByScalar(distanceToNeighbor * distanceToNeighbor, reverseDelta);
          add2Vectors(separationVel, reverseDelta);

          add2Vectors(centerOfMass, theirPos);
          numberOfNeighbors++;
      	}
      }
    }

    if (numberOfNeighbors > 0) {
      alignmentVelocity = divVectorByScalarNew(float(numberOfNeighbors), alignmentNumerator);

      divVectorByScalar(float(numberOfNeighbors), centerOfMass);


      float3 alignmentVelocityTemp = mulVectorByScalarNew(c_alignment, alignmentVelocity);

      float3 separationVelTemp = mulVectorByScalarNew(c_separation, separationVel);

      float3 deltaCenterTemp = sub2VectorsNew(centerOfMass, myPosition);
      mulVectorByScalar(c_cohesion, deltaCenterTemp);

      float3 deltaTargetPosTemp = sub2VectorsNew(target, myPosition);
      normalizeVector(deltaTargetPosTemp);
      mulVectorByScalar(c_seek, deltaTargetPosTemp);

      float3 sum = make_float3(0, 0, 0);
      add2Vectors(sum, alignmentVelocityTemp);
      add2Vectors(sum, separationVelTemp);
      add2Vectors(sum, deltaCenterTemp);
      add2Vectors(sum, deltaTargetPosTemp);

      desiredVel.x = sum.x;
      desiredVel.y = sum.y;
      desiredVel.z = sum.z;
    } else {
      float3 deltaPosTarget = sub2VectorsNew(target, myPosition);
      desiredVel = mulVectorByScalarNew(c_seek, deltaPosTarget);
    }
    if (magnitudeOfVector(myPosition) > 800.0f) {
      float3 neg_myPosition = make_float3(-myPosition.x, -myPosition.y, -myPosition.z);
      normalizeVector(neg_myPosition);
      desiredVel = neg_myPosition;
    }
    // Calc acc from steering vel
    float3 deltaVel = sub2VectorsNew(desiredVel, myVelocity);
    desiredVel = truncate(deltaVel, 2.0f);
    divVectorByScalar(pos[index].w, desiredVel);
    acc[index] = desiredVel;
  }
}

__global__
void updatePosition(int n, float dt, float4 *pos, float3 *vel, float3 *acc) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if( index < n ) {

    float3 temp = mulVectorByScalarNew(dt, acc[index]);
    float3 sumTemp = add2VectorsNew(vel[index], temp);
    normalizeVector(sumTemp);
    mulVectorByScalar(2.0f, sumTemp);

    vel[index].x += sumTemp.x;
    vel[index].y += sumTemp.y;
    vel[index].z += sumTemp.z;

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

__host__
void initCuda(int n) {
  //fprintf(stdout, "   Initializing CUDA.\n");
  dim3 fullBlocksPerGrid((int)ceil(float(n)/float(BlockSize)));

  checkCudaErrors( cudaMalloc((void**)&dev_pos, n*sizeof(float4)) );
  checkCudaErrors( cudaMalloc((void**)&dev_vel, n*sizeof(float3)) );
  checkCudaErrors( cudaMalloc((void**)&dev_acc, n*sizeof(float3)) );

  generateRandomPosArray<<<fullBlocksPerGrid, BlockSize>>>(n, dev_pos, boidMass);
  generateRandomVelArray<<<fullBlocksPerGrid, BlockSize>>>(n, dev_vel);
}

__host__
void cudaFlockingUpdateWrapper(int n, float dt, float3 target) {
  dim3 fullBlocksPerGrid((int)ceil(float(n)/float(BlockSize)));

  //fprintf(stdout, "   Updating Acceleration and position.\n");
  updateAccelaration<<<fullBlocksPerGrid, BlockSize>>>(n, dev_pos, dev_vel, dev_acc, target);
  updatePosition<<<fullBlocksPerGrid, BlockSize>>>(n, dt, dev_pos, dev_vel, dev_acc);

}

__host__
void cudaUpdateVBO(int n, float* vbodptr, float* velptr) {
  dim3 fullBlocksPerGrid((int)ceil(float(n)/float(BlockSize)));

  //fprintf(stdout, "   Sending changes to VBO.\n");
  sendToVBO<<<fullBlocksPerGrid, BlockSize>>>(n, dev_pos, dev_vel, vbodptr, velptr, scene_scale);
}
