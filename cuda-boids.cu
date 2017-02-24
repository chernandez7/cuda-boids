// C HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <ctype.h>
#include <assert.h>
#include <vector>
// OpenGL
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#endif
// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // interop functionality
#include <helper_cuda.h>
#include <helper_cuda_gl.h> // checkCudaErrors()
// COMMON
#include <aligned_allocator.h>
// Object Headers
#include "boid.h"
#include "flock.h"
#include "vector3f.h"

// Rotation of X-axis camera perspective
double rX = 0.0;
// Rotation of Y-axis camera perspective
double rY = 0.0;
// Time var to increment function over time
float sim_time = 0.0;
// Number of boids in simulation
int n = 0;
// Monitor metrics
const unsigned int window_width = 512;
const unsigned int window_height = 512;
// Mesh metrics
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;
// OpenGL Vertex Buffer Object
GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;

/* Kernel function sent to GPU that does calculations required in boid algorithm*/
__global__ void kernel(float4* positions, unsigned int width, unsigned int height, float t) {
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

   float u = x / (float)width;
   float v = y / (float)height;
   u = u * 2.0f - 1.0f;
   v = v * 2.0f - 1.0f;

   float freq = 4.0f;
   float w = sinf(u * freq + t)
	   * cosf(v * freq + t) * 0.5f;

   positions[y * width + x] = make_float4(u, w, v, 1.0f);
}

// Function called when incorrect command line syntax is called or -h flag is passed
__host__ void help() {
   fprintf(stderr,"./boids --help|-h --nboids|-n \n");
}

// Function to setup and launch kernel
__host__ void launchKernel(float4* positions, unsigned int width, unsigned int height, float t) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
  fprintf(stdout, "   launching kernel\n");
  kernel<<<dimGrid, dimBlock>>>(positions, width, height, t);
  fprintf(stdout, "   exiting kernel\n");
}

// Runs all CUDA related functions
__host__ void runCUDA(struct cudaGraphicsResource** vbo_resource) {
  fprintf(stdout, "   running cuda\n");
  // Map VBO to GL with CUDA
  float4* positions;
  // Map resource into GPU memory
  checkCudaErrors( cudaGraphicsMapResources(1, vbo_resource, 0) ); 
  size_t num_bytes;
  // Gets pointer to mapped resource
  checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void ** )&positions,
		  &num_bytes,
		  *vbo_resource) );
  // Launch kernel with pointer to mapped resource
  launchKernel(positions, mesh_width, mesh_height, sim_time);
  // Unmapping Buffer sends resource back to OpenGL accessible memory
  fprintf(stdout, "   unmapping buffer\n");
  checkCudaErrors( cudaGraphicsUnmapResources(1, vbo_resource, 0) );
  fprintf(stdout, "     x:%f  y:%f  z:%f w:%f\n", &positions[10].x, &positions[10].y, &positions[10].z, &positions[10].w);
}

__host__ void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags) {
  fprintf(stdout, "   creating vbo\n");
  assert(vbo);
  // Create vertex buffer object
  glGenBuffers(1, vbo);
  // Bind pointer to buffer as an GL_ARRAY_BUFFER type
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  unsigned int size = mesh_width * mesh_height * 4 * sizeof(float); // Size of allocated memory
  // Add actual data to GL Buffer 
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

  // Unbind Buffer
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Register VBO to CUDA memory
  checkCudaErrors( cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags) );
}

__host__ void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res) {
  fprintf(stdout, "   deleting vbo\n");
  // Unregister with CUDA
  checkCudaErrors( cudaGraphicsUnregisterResource(vbo_res) );

  // Delete VBO and free pointer
  glBindBuffer(1, *vbo);
  glDeleteBuffers(1, vbo);

  *vbo = 0;
}

// Main render function for GLUT which loops until exit
__host__ void Render() {

  // Launch Kernel
  runCUDA(&positionsVBO_CUDA);

  // Clear screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set View Matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Perspective modifications
  glRotatef(rX, 1.0, 0.0, 0.0);
  glRotatef(rY, 0.0, 1.0, 0.0);

  // Render from VBO
  glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  glColor3f(1.0f, 1.0f, 0.0f);
  fprintf(stdout, "   drawing vertices\n");
  glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height); // Rendering happens here
  glDisableClientState(GL_VERTEX_ARRAY);

  // Switch Buffers to show rendered
  glutSwapBuffers();

  // Increment Time
  sim_time += 0.05f;
  fprintf(stdout, "   simtime:%f\n", sim_time);
}

// Controls for simulation
__host__ void Keyboard(int key, int x, int y) {
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

// Idle function for GLUT, called when nothing is happening
__host__ void idleSim() {
  glutPostRedisplay();
}

// Called when OpenGL Window is resized to handle scaling
__host__ void windowResize(int height, int width) {
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (double)width / (double)height, 0.1, 10.0);
}

__host__ void GLInit(int argc, char* argv[]) {
  fprintf(stdout, "   initializing gl\n");
  // Create Window
  glutInit(&argc, argv);
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA	 | GLUT_DEPTH );
  glutInitWindowSize(window_width, window_height);
  glutInitWindowPosition(0, 0);
  glutCreateWindow("cuda-boids");

  // Set GLUT functions for simulation
  glutReshapeFunc(windowResize);
  glutDisplayFunc(Render);
  glutIdleFunc(idleSim);
  glutSpecialFunc(Keyboard);

  // Allow Depth and Colors
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_COLOR_MATERIAL);

  // Set the color of the background
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glDisable(GL_DEPTH_TEST);

  // Set perspective mode
  glViewport(0, 0, window_width, window_height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (double)mesh_width / (double)mesh_height, 1.0, 200.0);
}

__host__ void printDeviceProps() {
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
}

// De-allocation of Memory after GLUT stops
__host__ void onGLUTExit() {
   fprintf(stdout, "De-Allocating memory.\n");

   if (positionsVBO) {
     deleteVBO(&positionsVBO, positionsVBO_CUDA);
   }

   cudaDeviceReset();
}

__host__ int main (int argc, char* argv[]) {
   // Parse command line parameters
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

   // Print out GPU Details
   printDeviceProps();

   // OpenGL / GLUT Initialization
   GLInit(argc, argv);

   // Create VBO
   createVBO(&positionsVBO, &positionsVBO_CUDA, cudaGraphicsMapFlagsWriteDiscard);

   // Run initial CUDA step
   cudaGLSetGLDevice(0);
   //runCUDA(&positionsVBO_CUDA);

   // Start GLUT Loop
   fprintf(stdout, "   starting main loop\n");
   glutMainLoop();

   //onGLUTExit();

  return 0;
}

