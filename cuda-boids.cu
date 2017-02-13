#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <ctype.h>

#include <cuda_runtime.h>

#include <my_timer.h>
#include <aligned_allocator.h>

#include "boid.h"
#include "flock.h"
#include "vector3f.h"

static double device_time = 0.0;

__global__ void gpu_boids_kernel(int n, Flock* dev_flock) {
   //dev_flock.update();
}

void help() {
   fprintf(stderr,"./boids --help|-h --nboids|-n \n");
}

__host__ void gpu_boids(int n, Flock* h_flock) {
  // Allocate Device Memory
  Flock* dev_flock = NULL;
  cudaMalloc(&dev_flock, sizeof(Boid)*n);
  // Create Cuda Events
  cudaEvent_t calc1_event, calc2_event;
  cudaEventCreate(&calc1_event);
  cudaEventCreate(&calc2_event);
  // Copy Host Memory to Device
  cudaMemcpy(dev_flock, h_flock, sizeof(Boid)*n, cudaMemcpyHostToDevice);
  cudaEventRecord(calc1_event);
  // Entering Kernel
  fprintf(stdout,"entering kernel.\n");
  gpu_boids_kernel<<<1,1>>>(n, dev_flock);
  cudaEventRecord(calc2_event);
  // Free Device Memory
  cudaFree(dev_flock);

  // Record device time
  float time;
  cudaEventElapsedTime(&time, calc1_event, calc2_event);
  device_time += time;

  // Destroy event timers
  cudaEventDestroy(calc1_event);
  cudaEventDestroy(calc2_event);
}

int main (int argc, char* argv[]) {
   /* Define the number of boids. The default is 1000. */
   int n = 1000;

   for (int i = 1; i < argc; ++i) {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

      if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0) {
         help();
         return 1;
      }
      else if (strcmp(argv[i],"--nboids") == 0 || strcmp(argv[i],"-n") == 0) {
         check_index(i+1,"--nboids|-n");
         i++;
         if (isdigit(*argv[i]))
            n = atoi( argv[i] );
      } else {
         fprintf(stderr,"Unknown option %s\n", argv[i]);
         help();
         return 1;
      }
   }

   //  Memory Allocation
   fprintf(stdout, "Allocating memory for flock.\n");

   Flock* flock = NULL;
   Allocate(flock, sizeof(Boid)*n);

   {
      cudaDeviceProp props;
      cudaGetDeviceProperties( &props, 0 );
      fprintf(stderr, "   name:                           %s\n", props.name);
      fprintf(stderr, "   major.minor:                    %d.%d\n", props.major, props.minor);
      fprintf(stderr, "   totalGlobalMem:                 %d (MB)\n", props.totalGlobalMem / (1024*1024));
      fprintf(stderr, "   sharedMemPerBlock:              %d (KB)\n", props.sharedMemPerBlock / 1024);
      fprintf(stderr, "   sharedMemPerMultiprocessor:     %d (KB)\n", props.sharedMemPerMultiprocessor / 1024);
      fprintf(stderr, "   regsPerBlock:                   %d\n", props.regsPerBlock);
      fprintf(stderr, "   warpSize:                       %d\n", props.warpSize);
      fprintf(stderr, "   multiProcessorCount:            %d\n", props.multiProcessorCount);
      fprintf(stderr, "   maxThreadsPerBlock:             %d\n", props.maxThreadsPerBlock);
   }

   //double t_gpu = 0, t_host = 0;
   //myTimer_t t_start = getTimeStamp();

   gpu_boids(n, flock);
   //fprintf(stdout, device_time);  
   // Memory Deallocation
   fprintf(stdout, "De-Allocating memory.\n");
   Deallocate(flock);
   return 0;
}
