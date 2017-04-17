
#include "kernel.h"
#include <ctime>

dim3 threadsPerBlock(BlockSize);

const float boidMass = 1.0;
const float scene_scale = 4e1;
const __device__ float maxSpeed = 3.0;
const __device__ float sep_dist = 100000;
const __device__ float ali_dist = 100;
const __device__ float coh_dist = 100000;

float4* dev_pos;
float3* dev_vel;
float3* dev_acc;
float3* dev_results;
curandState_t* dev_states;

__device__
float distanceFormula(float3 myPos, float3 theirPos) {
	float dx = myPos.x - theirPos.x;
	float dy = myPos.y - theirPos.y;
	float dz = myPos.z - theirPos.z;

	float dist = sqrt(dx*dx + dy*dy + dz*dz);
	return dist;
}

__device__
float3 add2Vectors(float3 v1, float3 v2) {
	float3 temp = make_float3(v1.x, v1.y, v1.z);
	temp.x += v2.x;
	temp.y += v2.y;
	temp.z += v2.z;
	return temp;
}

__device__
float3 sub2Vectors(float3 v1, float3 v2) {
	float3 temp = make_float3(v1.x, v1.y, v1.z);
	temp.x -= v2.x;
	temp.y -= v2.y;
	temp.z -= v2.z;
	return temp;
}

__device__
float3 mulVectorByScalar(float scalar, float3 vector) {
	// Temp to not overwrite original vector
	float3 temp = make_float3(vector.x, vector.y, vector.z);
	temp.x *= scalar;
	temp.y *= scalar;
	temp.z *= scalar;
	return temp;
}

__device__
float3 divVectorByScalar(float scalar, float3 vector) {
	float3 temp = make_float3(vector.x, vector.y, vector.z);
	temp.x /= scalar;
	temp.y /= scalar;
	temp.z /= scalar;
	return temp;
}

__device__
float magnitudeOfVector(float3 vector) {
	return sqrt(
		vector.x * vector.x +
		vector.y * vector.y +
		vector.z * vector.z
	);
}

__device__
float3 normalizeVector(float3 vector) {
	float3 temp = make_float3(0, 0, 0);
	float magnitude = magnitudeOfVector(vector);
	if (magnitude > 0) {
		temp.x = vector.x / magnitude;
		temp.y = vector.y / magnitude;
		temp.z = vector.z / magnitude;
	}
	return temp;
}

__device__
float3 RNG(int nBoids, curandState_t* state, int max) {
	int threadID = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Copy state to local memory for efficiency
	curandState_t localState = state[threadID];
	float3 temp = make_float3(
		curand(&localState) % max,
		curand(&localState) % max,
		curand(&localState) % max
	);
	// Copy state back to global memory
	state[threadID] = localState;
	return temp;
}

__global__
void setupRNG(curandState_t* state, int seed) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	/* Each thread gets same seed, a different sequence number,
	no offset */
	curand_init(seed, index, 0, &state[index]);

}

__global__
void generateRandomPosArray(int n, curandState_t* states, float4* arr, float mass, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n) {

		float3 random = RNG(n, states, 1000);

		arr[index].x = random.x * scale;
		arr[index].y = random.y * scale;
		arr[index].z = random.z * scale;

		arr[index].w = mass;
	}
}

__global__
void generateRandomArray(int n, curandState_t* states, float3* arr) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n) {
		float3 random = RNG(n, states, 50);

		arr[index].x = random.x;
		arr[index].y = random.y;
		arr[index].z = random.z;
	}
}

__device__
float3 SeparationRule(int n, float4* pos, float3* vel, float3 target) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < n) {
		float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);
		float3 myVelocity = make_float3(vel[index].x, vel[index].y, vel[index].z);
		float3 steer = make_float3(0, 0, 0);
		float neighborDistance = sep_dist;
		int numberOfNeighbors = 0;

		// Check against all particles if in range

		for (int i = 0; i < n; i++) {
			float3 theirPos = make_float3(pos[i].x, pos[i].y, pos[i].z);
			float distanceToNeighbor = distanceFormula(myPosition, theirPos);
			//float distanceToNeighbor = distanceFormula(myPosition, target);
			// Add the difference of positions to steer force
			if (distanceToNeighbor > 0 && distanceToNeighbor < neighborDistance) {
				// calc and normalize delta
				float3 deltaPos = sub2Vectors(myPosition, theirPos);
				float3 normalized_delta = normalizeVector(deltaPos);

				// div delta by distance to weight
				normalized_delta.x /= distanceToNeighbor;
				normalized_delta.y /= distanceToNeighbor;
				normalized_delta.z /= distanceToNeighbor;

				// add delta to steer
				steer.x += normalized_delta.x;
				steer.y += normalized_delta.y;
				steer.z += normalized_delta.z;

				// increment number of neighbors
				numberOfNeighbors++;
			}
		}
		if (numberOfNeighbors > 0) {
			// weight steer by number of neighbors
			steer.x /= numberOfNeighbors;
			steer.y /= numberOfNeighbors;
			steer.z /= numberOfNeighbors;
		}
		float size = magnitudeOfVector(steer);
		if (size > 0) {
			float3 normalized_steer = normalizeVector(steer);

			//mulVectorByScalar(maxSpeed, normalized_steer);
			normalized_steer.x *= maxSpeed;
			normalized_steer.y *= maxSpeed;
			normalized_steer.z *= maxSpeed;

			//sub2Vectors(normalized_steer, myVelocity);
			normalized_steer.x -= myVelocity.x;
			normalized_steer.y -= myVelocity.y;
			normalized_steer.z -= myVelocity.z;

			//limit(0.5, normalized_steer);
			float steer_size = magnitudeOfVector(normalized_steer);
			if (steer_size > 0.5) {
				normalized_steer.x /= steer_size;
				normalized_steer.y /= steer_size;
				normalized_steer.z /= steer_size;
			}
			steer.x = normalized_steer.x;
			steer.y = normalized_steer.y;
			steer.z = normalized_steer.z;
		}
		return steer;
	}
	else {
		return make_float3(0, 0, 0);
	}
}

__device__
float3 AlignmentRule(int n, float4* pos, float3* vel) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < n) {
		float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);
		float3 myVelocity = make_float3(vel[index].x, vel[index].y, vel[index].z);

		float neighborDistance = ali_dist;
		int numberOfNeighbors = 0;
		float3 sum = make_float3(0, 0, 0);

		// Check against all particles if in range
		for (int i = 0; i < n; i++) {
			float3 theirPos = make_float3(pos[i].x, pos[i].y, pos[i].z);
			float3 theirVel = make_float3(vel[i].x, vel[i].y, vel[i].z);
			// Get distance between you and neighbor
			float distanceToNeighbor = distanceFormula(myPosition, theirPos);
			if (distanceToNeighbor > 0 && distanceToNeighbor < neighborDistance) {

				// add vel to sum
				sum.x += theirVel.x;
				sum.y += theirVel.y;
				sum.z += theirVel.z;

				numberOfNeighbors++;
			}
		}
		if (numberOfNeighbors > 0) {
			// div sum by # of neighbors
			sum.x /= numberOfNeighbors;
			sum.y /= numberOfNeighbors;
			sum.z /= numberOfNeighbors;

			float3 normalized_sum = normalizeVector(sum);

			//mulVectorByScalar(maxSpeed, normalized_sum);
			normalized_sum.x *= maxSpeed;
			normalized_sum.y *= maxSpeed;
			normalized_sum.z *= maxSpeed;

			// sub my vel from sum
			normalized_sum.x -= myVelocity.x;
			normalized_sum.y -= myVelocity.y;
			normalized_sum.z -= myVelocity.z;

			// limit vel
			float size = magnitudeOfVector(normalized_sum);
			if (size > 5) {
				normalized_sum.x /= size;
				normalized_sum.y /= size;
				normalized_sum.z /= size;
			}

			return normalized_sum;

		}
		else { // No neighbors nearby
			return make_float3(0, 0, 0);
		}
	}
	else { // index not in range
		return make_float3(0, 0, 0);
	}
}

__device__
float3 CohesionRule(int n, float4* pos, float3* vel) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < n) {

		float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);

		float neighborDistance = coh_dist;
		int numberOfNeighbors = 0;
		float3 sum = make_float3(0, 0, 0);

		// Check against all particles if in range
		for (int i = 0; i < n; i++) {
			float3 theirPos = make_float3(pos[i].x, pos[i].y, pos[i].z);
			// Get distance between you and neighbor
			float distanceToNeighbor = distanceFormula(myPosition, theirPos);
			// If in range add their pos to sum
			if (distanceToNeighbor > 0 && distanceToNeighbor < neighborDistance) {

				// add their pos to sum
				sum.x += theirPos.x;
				sum.y += theirPos.y;
				sum.z += theirPos.z;

				numberOfNeighbors++;
			}
		}
		if (numberOfNeighbors > 0) {
			//divVectorByScalar(numberOfNeighbors, sum);
			sum.x /= numberOfNeighbors;
			sum.y /= numberOfNeighbors;
			sum.z /= numberOfNeighbors;

			//return seek(sum, vel[index]);
			float3 desired = make_float3(0, 0, 0);
			//sub2Vectors(desired, sum);
			desired.x -= sum.x;
			desired.y -= sum.y;
			desired.z -= sum.z;

			float3 normalized_desired = normalizeVector(desired);

			//mulVectorByScalar(maxSpeed, desired);
			normalized_desired.x *= maxSpeed;
			normalized_desired.y *= maxSpeed;
			normalized_desired.z *= maxSpeed;

			// Steering = desired - vel
			//return limit(0.5, sub2VectorsNew(desired, vel));
			normalized_desired.x -= vel[index].x;
			normalized_desired.y -= vel[index].y;
			normalized_desired.z -= vel[index].z;

			float size = magnitudeOfVector(normalized_desired);
			if (size > 0.5) {
				normalized_desired.x /= size;
				normalized_desired.y /= size;
				normalized_desired.z /= size;
			}
			return normalized_desired;
		}
		else {
			// No neighbors nearby
			return make_float3(0, 0, 0);
		}
	}
	else {
		return make_float3(0, 0, 0);
	}
}

__global__
void flocking(int n, float4* pos, float3* vel, float3* acc, float3 target) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < n) {
		// Separation
		float3 separation = SeparationRule(n, pos, vel, target);
		// Alignment
		float3 alignment = AlignmentRule(n, pos, vel);
		// Cohesion
		float3 cohesion = CohesionRule(n, pos, vel);

		// Apply Arbitrary Weights
		separation.x *= 1.5;
		separation.y *= 1.5;
		separation.z *= 1.5;

		alignment.x *= 1.0;
		alignment.y *= 1.0;
		alignment.z *= 1.0;

		cohesion.x *= 1.0;
		cohesion.y *= 1.0;
		cohesion.z *= 1.0;

		// Apply Forces to acc
		acc[index].x += separation.x;
		acc[index].x += alignment.x;
		acc[index].x += cohesion.x;

		acc[index].y += separation.y;
		acc[index].y += alignment.y;
		acc[index].y += cohesion.y;

		acc[index].z += separation.z;
		acc[index].z += alignment.z;
		acc[index].z += cohesion.z;
	}
}

__global__
void updatePosition(int n, float dt, float4 *pos, float3 *vel, float3 *acc, int window_width, int window_height) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < n) {

		// Done so slow down won't be as abrupt
		// Multiply acc by scalar dt to not have an instant stop
		acc[index].x *= dt;
		acc[index].y *= dt;
		acc[index].z *= dt;

		// Update velocity by adding acceleration
		vel[index].x += acc[index].x;
		vel[index].y += acc[index].y;
		vel[index].z += acc[index].z;

		// Limit speed if exceeding maxSpeed
		float size = magnitudeOfVector(vel[index]);
		if (size > maxSpeed) {
			//divVectorByScalar(size, vector);
			vel[index].x /= size;
			vel[index].y /= size;
			vel[index].z /= size;
		}

		/*** EULER METHOD ***
		pos[index].x += vel[index].x;
		pos[index].y += vel[index].y;
		pos[index].z += vel[index].z;
		*/

		/*** RK4 METHOD ***/

		// k1
		float3 k1 = vel[index];

		// k2
		float3 k2 = make_float3(0, 0, 0);
		k2.x = k1.x + 0.5f * dt * k1.x;
		k2.y = k1.y + 0.5f * dt * k1.y;
		k2.z = k1.z + 0.5f * dt * k1.z;

		// k3
		float3 k3 = make_float3(0, 0, 0);
		k3.x = k1.x + 0.5f * dt * k2.x;
		k3.y = k1.y + 0.5f * dt * k2.y;
		k3.z = k1.z + 0.5f * dt * k2.z;

		// k4
		float3 k4 = make_float3(0, 0, 0);
		k4.x = k1.x + dt * k3.x;
		k4.y = k1.y + dt * k3.y;
		k4.z = k1.z + dt * k3.z;

		// increment
		float3 increment = make_float3(0,0,0);
		increment.x = 1.0f / 6.0f * (k1.x + 2.0f * k2.x + 2.0f * k3.x + k4.x);
		increment.y = 1.0f / 6.0f * (k1.y + 2.0f * k2.y + 2.0f * k3.y + k4.y);
		increment.z = 1.0f / 6.0f * (k1.z + 2.0f * k2.z + 2.0f * k3.z + k4.z);

		// update pos
		pos[index].x += increment.x * dt;
		pos[index].y += increment.y * dt;
		pos[index].z += increment.z * dt;

		// Reset acc to 0
		acc[index].x = 0;
		acc[index].y = 0;
		acc[index].z = 0;

		// Fix positions if particle is off screen
		if (pos[index].x < 0)				pos[index].x += window_width;
		if (pos[index].y < 0)				pos[index].y += window_height;
		if (pos[index].x > window_width)	pos[index].x -= window_width;
		if (pos[index].y > window_height)	pos[index].y -= window_height;
	}
}

__global__
void sendToVBO(int n, float scale, float4* pos, float3* vel, float3* acc,
	float* posVBO, float* velVBO, float* accVBO) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float divisor = 2.0f / scale;

	if (index < n) {
		posVBO[4 * index + 0] = pos[index].x / divisor;
		posVBO[4 * index + 1] = pos[index].y / divisor;
		posVBO[4 * index + 2] = pos[index].z / divisor;
		posVBO[4 * index + 3] = 1;

		velVBO[3 * index + 0] = vel[index].x / divisor;
		velVBO[3 * index + 1] = vel[index].y / divisor;
		velVBO[3 * index + 2] = vel[index].z / divisor;

		accVBO[3 * index + 0] = acc[index].x / divisor;
		accVBO[3 * index + 1] = acc[index].y / divisor;
		accVBO[3 * index + 2] = acc[index].z / divisor;
	}
}

/*****************************************************************
 *
 *	Wrapper Functions
 *
 ****************************************************************/

__host__
void initCuda(int n) {
	//fprintf(stdout, "   Initializing CUDA.\n");
	dim3 fullBlocksPerGrid((int)ceil(float(n) / float(BlockSize)));

	// Malloc's
	cudaMalloc((void**)&dev_pos, n * sizeof(float4));
	cudaMalloc((void**)&dev_vel, n * sizeof(float3));
	cudaMalloc((void**)&dev_acc, n * sizeof(float3));

	cudaMalloc((void **)&dev_results, n * sizeof(float3));
	//cudaMemset(dev_results, 0, n * sizeof(float));
	cudaMalloc((void **)&dev_states, n * sizeof(curandState_t));

	// Kernels
	setupRNG << <fullBlocksPerGrid, BlockSize >> > (dev_states, time(NULL));
	generateRandomPosArray << <fullBlocksPerGrid, BlockSize >> >(n, dev_states, dev_pos, boidMass, scene_scale);
	generateRandomArray << <fullBlocksPerGrid, BlockSize >> >(n, dev_states, dev_vel);
	generateRandomArray << <fullBlocksPerGrid, BlockSize >> >(n, dev_states, dev_acc);
}

__host__
void flock(int n, int window_width, int window_height, float3 target) {
	dim3 fullBlocksPerGrid((int)ceil(float(n) / float(BlockSize)));

	//fprintf(stdout, "   Updating Acceleration and position.\n");
	flocking << <fullBlocksPerGrid, BlockSize >> >(n, dev_pos, dev_vel, dev_acc, target);
	updatePosition << <fullBlocksPerGrid, BlockSize >> >(n, 0.5, dev_pos, dev_vel, dev_acc, window_width, window_height);

}

__host__
void cudaUpdateVBO(int n, float* vbodptr, float* velptr, float* accptr) {
	dim3 fullBlocksPerGrid((int)ceil(float(n) / float(BlockSize)));

	//fprintf(stdout, "   Sending changes to VBO.\n");
	sendToVBO << <fullBlocksPerGrid, BlockSize >> >(n, scene_scale, dev_pos, dev_vel, dev_acc, vbodptr, velptr, accptr);
}
