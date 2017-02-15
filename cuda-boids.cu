#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <ctype.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#endif
#include <cuda_runtime.h>

//#include <my_timer.h>
#include <aligned_allocator.h>

#include "boid.h"
#include "flock.h"
#include "vector3f.h"

static double device_time = 0.0;
double rX = 0.0;
double rY = 0.0;

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

void drawBoid() {
  glClearColor(0.4, 0.4, 0.4, 0.4);
  glClear(GL_COLOR_BUFFER_BIT);
  glColor3f(1.0, 1.0, 1.0);
  glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

  glBegin(GL_TRIANGLES);
    glVertex3f(-0.7, 0.7, 0);
    glVertex3f(0.7, 0.7, 0);
    glVertex3f(0, -1, 0);
  glEnd();

  glFlush();
  glutSwapBuffers();
}

void Render() {
  // Set Background Color
  glClearColor(0.4, 0.4, 0.4, 1.0);
  // Clear screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Reset transformations
  glLoadIdentity();
  
  // Perspective modifications
  glRotatef(rX, 1.0, 0.0, 0.0);
  glRotatef(rY, 0.0, 1.0, 0.0);

  drawBoid();
}

void Keyboard(int key, int x, int y) {
  if (key == GLUT_KEY_RIGHT) {
    rY += 15;
  } else if (key == GLUT_KEY_LEFT) {
    rY -= 15;
  } else if (key == GLUT_KEY_DOWN) {
    rX -= 15;
  } else if (key == GLUT_KEY_UP) {
    rX += 15;
  }

  // Request display update
  glutPostRedisplay();		  
}

void GLInit(int argc, char* argv[]) {
  glutInit(&argc, argv);
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
  glutInitWindowSize(700, 700);
  glutInitWindowPosition(20, 20);
  glutCreateWindow("cuda-boids");
}

// Called when OpenGL Window is resized to handle scaling
void windowResize(int height, int width) {
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (double)width / (double)height, 1.0, 200.0);
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
      fprintf(stderr, "   totalGlobalMem:                 %lu (MB)\n", props.totalGlobalMem / (1024*1024));
      fprintf(stderr, "   sharedMemPerBlock:              %lu (KB)\n", props.sharedMemPerBlock / 1024);
      fprintf(stderr, "   sharedMemPerMultiprocessor:     %lu (KB)\n", props.sharedMemPerMultiprocessor / 1024);
      fprintf(stderr, "   regsPerBlock:                   %d\n", props.regsPerBlock);
      fprintf(stderr, "   warpSize:                       %d\n", props.warpSize);
      fprintf(stderr, "   multiProcessorCount:            %d\n", props.multiProcessorCount);
      fprintf(stderr, "   maxThreadsPerBlock:             %d\n", props.maxThreadsPerBlock);
   }

   //double t_gpu = 0, t_host = 0;
   //myTimer_t t_start = getTimeStamp();
  
   // OpenGL / GLUT
   GLInit(argc, argv);
   glutReshapeFunc(windowResize);
   glutDisplayFunc(Render);
   glutSpecialFunc(Keyboard);

   glutMainLoop();

   gpu_boids(n, flock);
   //fprintf(stdout, device_time);  
   // Memory Deallocation
   fprintf(stdout, "De-Allocating memory.\n");
   Deallocate(flock);
   return 0;
}

