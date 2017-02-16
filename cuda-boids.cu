#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <ctype.h>
#include <vector>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h> 

#include <aligned_allocator.h>

#include "boid.h"
#include "flock.h"
#include "vector3f.h"

double rX = 0.0;
double rY = 0.0;
float time = 0.0;
int n = 0;
int window_width = 1000;
wint window_height = 1000;
int mesh_width = 500;
int msh_height = 500;
GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;

__global__ void kernel(float4* positions, unsigned int mesh_width, unsigned int mesh_height, float time) {
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

   float u = x / (float)mesh_width;
   float v = y / (float)mesh_height;
   u = u * 2.0f - 1.0f;
   v = v * 2.0f - 1.0f;

   float freq = 4.0f;
   float w = sinf(u * freq + time)
	   * cosf(v * freq + time) * 0.5f;

   positions[y * width + x] = make_float4(u, w, v, 1.0f);
}

__host__ void help() {
   fprintf(stderr,"./boids --help|-h --nboids|-n \n");
}

__host__ void launchKernel(float4* positions, unsigned int mesh_width, unsigned int mesh_height, float time) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
  fprintf(stdout, "   Entering Kernel.\n");
  kernel<<<dimGrid, dimBlock>>>(positions, width, height, time);
  fprintf(stdout, "   Exited Kernel.\n");
}

__host__ void drawBoid() {
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

__host__ void Render() {
  
  // Clear screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Perspective modifications
  glRotatef(rX, 1.0, 0.0, 0.0);
  glRotatef(rY, 0.0, 1.0, 0.0);
  
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glColor3f(1.0, 0.0, 0.0);
  glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

  glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, width * height); // Where the magic happens
  glDisableClientState(GL_VERTEX_ARRAY);
  
  // Switch Buffers
  glutSwapBuffers();
  glutPostRedisplay();
}

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

__host__ void GLInit(int argc, char* argv[]) {

  // Create Window
  glutInit(&argc, argv);
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
  glutInitWindowSize(window_width, window_height);
  glutInitWindowPosition(0, 0);
  glutCreateWindow("cuda-boids");

  glutReshapeFunc(windowResize);
  glutDisplayFunc(Render);
  glutSpecialFunc(Keyboard);

  // Allow Depth and Colors
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_COLOR_MATERIAL);

  // Set the color of the background
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glDisable(GL_DEPTH_TEST);
  glViewport(0, 0, window_width, window_height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (double)width / (double)height, 1.0, 200.0);
}

// Called when OpenGL Window is resized to handle scaling
__host__ void windowResize(int height, int width) {
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (double)width / (double)height, 1.0, 200.0);
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
void onGLUTExit() {
   fprintf(stdout, "De-Allocating memory.\n");
   //Deallocate(h_flock);
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags) {
  
  // Create buffer object and register it with CUDA
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  
  // Unbind Buffer
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  // Register VBO
  checkCudaErrors( cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags) );
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res) {
  // Unregister with CUDA
  checkCudaErrors( cudaGraphicsUnregisterResource(vbo_res) );

  glBindBuffer(1, *vbo);
  glDeleteBuffers(1, vbo);
}

void runCUDA(struct cudaGraphicsResource ** vbo_resource) {
  // Map VBO to GL with CUDA
  float4* positions;
  checkCudaErrors( cudaGraphicsMapResources(1, vbo_resource, 0) );
  size_t num_bytes;
  checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void ** )&positions,
		  &num_bytes,
		  vbo_resource) );

  launchKernel(positions, mesh_width, mesh_height, time); 
  // Unmap Buffer
  checkCudaErrors( cudaGraphicsUnmapResources(1, vbo_resource, 0) );
}

__host__ int main (int argc, char* argv[]) {

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
   createVBO(&positionsVBO, positionsVBO_CUDA, cudaGraphicsMapFlagsWriteDiscard);

   // Run initial CUDA step
   checkCudaErrors( cudaGLSetGLDevice(0) );
   runCUDA(&positionsVBO_CUDA); 
   
   // Start GLUT Loop
   glutMainLoop();
   
   //onGLUTExit();

  return 0;
}

