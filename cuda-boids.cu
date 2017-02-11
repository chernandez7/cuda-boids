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

__global__ void gpu_boids_kernel() {
  
}

//static double device_time = 0.0;

void help() {
   fprintf(stderr,"./boids --help|-h --nboids|-n \n");
}

int main (int argc, char* argv[]) {
   /* Define the number of boids. The default is 1000. /
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

   //myTimer_t t_start = getTimeStamp();
   gpu_boids_kernel<<<1, 1>>>();


   // Memory Deallocation
*/
   fprintf(stdout,"entering kernel.\n");
   gpu_boids_kernel<<<1,1>>>();
   return 0;
}

