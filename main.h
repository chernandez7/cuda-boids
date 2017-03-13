
#ifndef MAIN_H
#define MAIN_H

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

GLuint positionVBO = (GLuint)NULL;
GLuint velocityVBO = (GLuint)NULL;
GLuint IBO = (GLuint)NULL;
GLuint displayImage;

// Rotation of X-axis camera perspective
double rX = 0.0;
// Rotation of Y-axis camera perspective
double rY = 0.0;
int window_width = 1000;
int window_height = 1000;
int timeSinceLastFrame;
int mouse_old_x, mouse_old_y;
int nBoids = 0;
float viewPhi = 0;
float viewTheta = 0;
float3 seekTarget = make_float3(0, 0, 0);

int main(int argc, char* argv[]);
void printDeviceProps();
void Init(int argc, char* argv[]);
void initVAO();
void idleSim();
void windowResize(int height, int width);
void Keyboard(unsigned char key, int x, int y);
void help();
void Render();
void runCUDA();
void mouseMotion(int x, int y);

#endif
