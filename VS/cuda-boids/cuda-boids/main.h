#ifndef MAIN_H
#define MAIN_H

// C HEADERS
#ifdef _WIN32
#include <Windows.h>
#endif
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

GLuint positionLocation = 0;
GLuint velocityLocation = 1;
GLuint accelerationLocation = 2;

GLuint positionVBO = (GLuint)NULL;
GLuint velocityVBO = (GLuint)NULL;
GLuint accelerationVBO = (GLuint)NULL;
GLuint IBO = (GLuint)NULL;

GLuint program[2];
const unsigned int PASS_THROUGH = 1;
const char *attributeLocations[] = { "position", "velocity", "acceleration"};

glm::mat4 projection;
glm::mat4 view;
glm::vec3 cameraPosition(3.5f, 3.5f, 3.0f);

float fovy = 90.0f;
float zNear = 0.10f;
float zFar = 1000.0;

int timebase = 0;
int frame = 0;

int window_width = 1000;
int window_height = 1000;
int timeSinceLastFrame;
int mouse_old_x, mouse_old_y;

int nBoids = 0;
bool followMouse = false;
bool naive = false;
float viewPhi = 0;
float viewTheta = 0;
float3 seekTarget;

float sep_dist = 100;
float ali_dist = 400;
float coh_dist = 300;

float sep_weight = 1.0f;
float ali_weight = 1.5f;
float coh_weight = 1.0f;


int main(int argc, char* argv[]);
void printDeviceProps();
void printControls();
void initVAO();
void windowResize(int height, int width);
void Keyboard(unsigned char key, int x, int y);
void help();
void Render();
void mouseMotion(int x, int y);
void Init(int argc, char* argv[]);
void runCUDA(bool followMouse, float sep_dist, float sep_weight, float ali_dist, float ali_weight, float coh_dist, float coh_weight);
void initShaders(GLuint * program);


#endif