#ifndef MAIN_H
#define MAIN_H

// C HEADERS
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
//#include <strings.h>
#include <ctype.h>
#include <assert.h>
#include <vector>
// OpenGL
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include "glslUtility.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#endif
// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // interop functionality
#include <helper_cuda.h>
#include <helper_cuda_gl.h> // checkCudaErrors()
// COMMON
//#include <aligned_allocator.h>

GLuint positionLocation = 1;
GLuint velocityLocation = 2;
GLuint accelerationLocation = 3;

GLuint positionVBO = (GLuint)NULL;
GLuint velocityVBO = (GLuint)NULL;
GLuint accelerationVBO = (GLuint)NULL;
GLuint IBO = (GLuint)NULL;

GLuint program[2];
const unsigned int PASS_THROUGH = 1;
const char *attributeLocations[] = { "position", "velocity", "acceleration" };

glm::mat4 projection;
glm::mat4 view;
glm::vec3 cameraPosition(3.5, 3.5, 3);

float fovy = 20.0f;
float zNear = 0.01f;
float zFar = 100.0;
int timebase = 0;
int frame = 0;


int window_width = 750;
int window_height = 750;
int timeSinceLastFrame;
int mouse_old_x, mouse_old_y;

int nBoids = 0;
float viewPhi = 0;
float viewTheta = 0;
float3 seekTarget;


int main(int argc, char* argv[]);
void printDeviceProps();
void initVAO();
void idleSim();
void windowResize(int height, int width);
void Keyboard(unsigned char key, int x, int y);
void help();
void Render();
void mouseMotion(int x, int y);
void Init(int argc, char* argv[]);
void runCUDA();
void initShaders(GLuint * program);


#endif
