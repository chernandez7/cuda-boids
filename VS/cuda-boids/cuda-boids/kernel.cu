
#include "kernel.h"
#include <ctime>

/*** Globals ***/
dim3 threadsPerBlock(BlockSize);

// Results from flock
float4* dev_pos;
float3* dev_vel;
float3* dev_acc;

float3* dev_sep;
float3* dev_ali;
float3* dev_coh;

// Results for RNG
float3* dev_pos_results;
float3* dev_vel_results;
float3* dev_acc_results;

// States for RNG
curandState_t* dev_states;

/*****************************************************************
*
*	Vector Functions
*
****************************************************************/

__device__
float distanceFormula(float3 myPos, float3 theirPos) {
	float dx = myPos.x - theirPos.x;
	float dy = myPos.y - theirPos.y;
	float dz = myPos.z - theirPos.z;

	float dist = sqrt(dx*dx + dy*dy + dz*dz);
	return dist;
}

__device__
float distanceFormula(float4 myPos, float3 theirPos) {
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
	float3 temp = make_float3(vector.x, vector.y, vector.z);
	float magnitude = magnitudeOfVector(temp);
	if (magnitude > 0) {
		temp.x /= magnitude;
		temp.y /= magnitude;
		temp.z /= magnitude;
	}
	return temp;
}

__device__
float3 RNG(int nBoids, curandState_t* state, float3* results, int max) {
	int threadID = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Copy state to local memory for efficiency
	curandState_t localState = state[threadID];

	results[threadID] = make_float3(
		curand(&localState) % max, 
		curand(&localState) % max, 
		curand(&localState) % max
	);
	// Copy state back to global memory
	state[threadID] = localState;

	return results[threadID];
}

/*****************************************************************
*
*	Kernels
*
****************************************************************/

__global__
void setupRNG(int n, curandState_t* state, int seed) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n) {
		// Each thread gets same seed, a different sequence number
		curand_init(seed, index, 0, &state[index]);
	}
}

__global__
void generateRandomPosArray(int n, curandState_t* states, float4* arr, float3* results, float mass, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < n) {
				
		float3 random = RNG(n, states, results, 2);

		arr[index].x = random.x * scale;
		arr[index].y = random.y * scale;
		arr[index].z = random.z * scale;

		arr[index].w = mass;
	}
}

__global__
void generateRandomArray(int n, curandState_t* states, float3* arr, float3* results) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n) {

		float3 random = RNG(n, states, results, 2);

		arr[index].x = random.x;
		arr[index].y = random.y;
		arr[index].z = random.z;
	}
}

__global__
void copyParametersKernel(float sep_dist, float sep_weight, float ali_dist, float ali_weight, float coh_dist, float coh_weight) {
	dev_sep_dist = sep_dist;
	dev_ali_dist = ali_dist;
	dev_coh_dist = coh_dist;

	dev_sep_weight = sep_weight;
	dev_ali_weight = ali_weight;
	dev_coh_weight = coh_weight;
}

__global__
void SeparationKernel(int n, float4* pos, float3* vel, float3 target, bool followMouse, float3* separation) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);
	float3 myVelocity = make_float3(vel[index].x, vel[index].y, vel[index].z);
	float3 steer = make_float3(0, 0, 0);
	float neighborDistance = dev_sep_dist; // Radius of consideration
	int numberOfNeighbors = 0;

	// Check against all particles if in range
	float distanceToNeighbor;
	for (int i = 0; i < n; i++) {
		float3 theirPos = make_float3(pos[i].x, pos[i].y, pos[i].z);

		if (followMouse) {
			distanceToNeighbor = distanceFormula(myPosition, target); // Follow Mouse

																	  // Add the difference of positions to steer force
			if (distanceToNeighbor > 0) {
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
		else {
			distanceToNeighbor = distanceFormula(myPosition, theirPos); // Follow Flock

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
		normalized_steer.x *= maxVelocity;
		normalized_steer.y *= maxVelocity;
		normalized_steer.z *= maxVelocity;

		//sub2Vectors(normalized_steer, myVelocity);
		normalized_steer.x -= myVelocity.x;
		normalized_steer.y -= myVelocity.y;
		normalized_steer.z -= myVelocity.z;

		//limit(0.5, normalized_steer);
		float steer_size = magnitudeOfVector(normalized_steer);
		if (steer_size > maxSteer) {
			normalized_steer.x /= steer_size;
			normalized_steer.y /= steer_size;
			normalized_steer.z /= steer_size;
		}

		steer.x = normalized_steer.x;
		steer.y = normalized_steer.y;
		steer.z = normalized_steer.z;
	}
	//return steer;
	separation[index].x = steer.x;
	separation[index].y = steer.y;
	separation[index].z = steer.z;
}

__global__
void AlignmentKernel(int n, float4* pos, float3* vel, float3* alignment) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);
	float3 myVelocity = make_float3(vel[index].x, vel[index].y, vel[index].z);

	float neighborDistance = dev_ali_dist; // Radius of consideration
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
		normalized_sum.x *= maxVelocity;
		normalized_sum.y *= maxVelocity;
		normalized_sum.z *= maxVelocity;

		// sub my vel from sum
		normalized_sum.x -= myVelocity.x;
		normalized_sum.y -= myVelocity.y;
		normalized_sum.z -= myVelocity.z;

		// limit vel
		float size = magnitudeOfVector(normalized_sum);
		if (size > 2.0f) {
			normalized_sum.x /= size;
			normalized_sum.y /= size;
			normalized_sum.z /= size;
		}

		//return normalized_sum;
		alignment[index].x = normalized_sum.x;
		alignment[index].y = normalized_sum.y;
		alignment[index].z = normalized_sum.z;

	}
	else { // No neighbors nearby
		   //return make_float3(0, 0, 0);
		alignment[index].x = 0;
		alignment[index].y = 0;
		alignment[index].z = 0;
	}
}

__global__
void CohesionKernel(int n, float4* pos, float3* vel, float3* cohesion) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float3 myPosition = make_float3(pos[index].x, pos[index].y, pos[index].z);

	float neighborDistance = dev_coh_dist; // Radius of consideration
	int numberOfNeighbors = 0;
	float3 sum = make_float3(0, 0, 0);

	// Check against all particles if in range
	for (int i = 0; i < n; i++) {
		float3 theirPos = make_float3(pos[i].x, pos[i].y, pos[i].z);
		// Get distance between you and neighbor
		float distanceToNeighbor = distanceFormula(myPosition, theirPos) + pos[i].w; // take into account boidMass
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

		float3 desired = make_float3(0, 0, 0);

		//sub2Vectors(desired, sum);
		desired.x -= sum.x;
		desired.y -= sum.y;
		desired.z -= sum.z;

		float3 normalized_desired = normalizeVector(desired);

		//mulVectorByScalar(maxSpeed, desired);
		normalized_desired.x *= maxVelocity;
		normalized_desired.y *= maxVelocity;
		normalized_desired.z *= maxVelocity;

		// Steering = desired - vel
		normalized_desired.x -= vel[index].x;
		normalized_desired.y -= vel[index].y;
		normalized_desired.z -= vel[index].z;

		// limit desired vel
		float size = magnitudeOfVector(normalized_desired);
		if (size > 0.5) {
			normalized_desired.x /= size;
			normalized_desired.y /= size;
			normalized_desired.z /= size;
		}
		//return normalized_desired;
		cohesion[index].x = normalized_desired.x;
		cohesion[index].y = normalized_desired.y;
		cohesion[index].z = normalized_desired.z;
	}
	else { // No neighbors nearby
		   //return make_float3(0, 0, 0);
		cohesion[index].x = 0;
		cohesion[index].y = 0;
		cohesion[index].z = 0;
	}
}

__global__
void flocking(int n, float4* pos, float3* vel, float3* acc, float3 target, bool followMouse, float3* separation, float3* alignment, float3* cohesion) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < n) {
		
		// Apply Arbitrary Weights
		separation[index].x *= dev_sep_weight;
		separation[index].y *= dev_sep_weight;
		separation[index].z *= dev_sep_weight;

		alignment[index].x *= dev_ali_weight;
		alignment[index].y *= dev_ali_weight;
		alignment[index].z *= dev_ali_weight;

		cohesion[index].x *= dev_coh_weight;
		cohesion[index].y *= dev_coh_weight;
		cohesion[index].z *= dev_coh_weight;
		
		// Apply Forces to acc
		acc[index].x += separation[index].x;
		acc[index].x += alignment[index].x;
		acc[index].x += cohesion[index].x;

		acc[index].y += separation[index].y;
		acc[index].y += alignment[index].y;
		acc[index].y += cohesion[index].y;

		acc[index].z += separation[index].z;
		acc[index].z += alignment[index].z;
		acc[index].z += cohesion[index].z;
	}
}

__global__
void updatePosition(int n, float dt, float4 *pos, float3 *vel, float3 *acc, int window_width, int window_height, bool naive) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < n) {

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
		if (size > maxVelocity) {
			//divVectorByScalar(size, vector);
			vel[index].x /= size;
			vel[index].y /= size;
			vel[index].z /= size;
		}

		vel[index].x *= maxVelocity;
		vel[index].y *= maxVelocity;
		vel[index].z *= maxVelocity;

		if (naive) {
			//** EULER METHOD ***
			pos[index].x += vel[index].x;
			pos[index].y += vel[index].y;
			pos[index].z += vel[index].z;	
		}
		else {

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
			float3 increment = make_float3(0, 0, 0);
			increment.x = 1.0f / 6.0f * (k1.x + 2.0f * k2.x + 2.0f * k3.x + k4.x);
			increment.y = 1.0f / 6.0f * (k1.y + 2.0f * k2.y + 2.0f * k3.y + k4.y);
			increment.z = 1.0f / 6.0f * (k1.z + 2.0f * k2.z + 2.0f * k3.z + k4.z);

			// update pos
			pos[index].x += increment.x * dt;
			pos[index].y += increment.y * dt;
			pos[index].z += increment.z * dt;
		}

		// Reset acc to 0
		acc[index].x = 0;
		acc[index].y = 0;
		acc[index].z = 0;

		float3 origin = make_float3(0, 0, 0);
		float deltaFromOrigin = distanceFormula(pos[index], origin);

		// Fix positions if particle is off screen
		if (deltaFromOrigin > 1500) {
			vel[index].x = -vel[index].x; // turn them around
			vel[index].y = -vel[index].y;
			vel[index].z = -vel[index].z;
		}

	}
}

__global__
void sendToVBO(int n, float scale, float4* pos, float3* vel, float3* acc,
	float* posVBO, float* velVBO, float* accVBO) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float scalar = 2.0f / scale;

	if (index < n) {
		posVBO[4 * index + 0] = pos[index].x * scalar;
		posVBO[4 * index + 1] = pos[index].y * scalar;
		posVBO[4 * index + 2] = pos[index].z * scalar;
		posVBO[4 * index + 3] = 1;

		velVBO[3 * index + 0] = vel[index].x * scalar;
		velVBO[3 * index + 1] = vel[index].y * scalar;
		velVBO[3 * index + 2] = vel[index].z * scalar;

		accVBO[3 * index + 0] = acc[index].x * scalar;
		accVBO[3 * index + 1] = acc[index].y * scalar;
		accVBO[3 * index + 2] = acc[index].z * scalar;
	}
}

/*****************************************************************
 *
 *	Wrapper Functions
 *	
 ****************************************************************/

__host__
void initCuda(int n) {
	fprintf(stdout, "   Initializing CUDA.\n\n");
	dim3 fullBlocksPerGrid((int)ceil(float(n) / float(BlockSize)));

	// Malloc's
	cudaMalloc((void**)&dev_pos, n * sizeof(float4));
	cudaMalloc((void**)&dev_vel, n * sizeof(float3));
	cudaMalloc((void**)&dev_acc, n * sizeof(float3));

	cudaMalloc((void**)&dev_sep, n * sizeof(float3));
	cudaMalloc((void**)&dev_ali, n * sizeof(float3));
	cudaMalloc((void**)&dev_coh, n * sizeof(float3));

	cudaMalloc((void **)&dev_pos_results, n * sizeof(float3));
	cudaMalloc((void **)&dev_vel_results, n * sizeof(float3));
	cudaMalloc((void **)&dev_acc_results, n * sizeof(float3));

	cudaMalloc((void **)&dev_states, n * sizeof(curandState_t));

	// Setup Kernels
	setupRNG << <fullBlocksPerGrid, BlockSize >> > (n, dev_states, time(NULL));
	generateRandomPosArray << <fullBlocksPerGrid, BlockSize >> >(n, dev_states, dev_pos, dev_pos_results, boidMass, scene_scale);
	generateRandomArray << <fullBlocksPerGrid, BlockSize >> >(n, dev_states, dev_vel, dev_vel_results);
	generateRandomArray << <fullBlocksPerGrid, BlockSize >> >(n, dev_states, dev_acc, dev_acc_results);
}

__host__
void flock(int n, int window_width, int window_height, float3 target, bool followMouse, bool naive, float sep_dist, float sep_weight, float ali_dist, float ali_weight, float coh_dist, float coh_weight) {
	dim3 fullBlocksPerGrid((int)ceil(float(n) / float(BlockSize)));

	copyParametersKernel << <fullBlocksPerGrid, BlockSize >> > (sep_dist, sep_weight, ali_dist, ali_weight, coh_dist, coh_weight);

	SeparationKernel <<<fullBlocksPerGrid, BlockSize>>>(n, dev_pos, dev_vel_results, target, followMouse, dev_sep);
	AlignmentKernel << <fullBlocksPerGrid, BlockSize >> >(n, dev_pos, dev_vel_results, dev_ali);
	CohesionKernel << <fullBlocksPerGrid, BlockSize >> >(n, dev_pos, dev_vel_results, dev_coh);

	flocking << <fullBlocksPerGrid, BlockSize >> >(n, dev_pos, dev_vel, dev_acc, target, followMouse, dev_sep, dev_ali, dev_coh);
	updatePosition << <fullBlocksPerGrid, BlockSize >> >(n, 0.5, dev_pos, dev_vel, dev_acc, window_width, window_height, naive);
}

__host__
void cudaUpdateVBO(int n, float* vbodptr, float* velptr, float* accptr) {
	dim3 fullBlocksPerGrid((int)ceil(float(n) / float(BlockSize)));

	sendToVBO << <fullBlocksPerGrid, BlockSize >> >(n, scene_scale, dev_pos, dev_vel, dev_acc, vbodptr, velptr, accptr);
}
