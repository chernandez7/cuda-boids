
#include "main.h"
#include "kernel.h"

__host__
int main(int argc, char* argv[]) {

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
           nBoids = atoi( argv[i] );
     } else {
        fprintf(stderr,"Unknown option %s\n", argv[i]);
        help();
        return 1;
     }
  }

  // OpenGL / GLUT Initialization
  Init(argc, argv);
  cudaGLSetGLDevice(0);

  // Register VBO
  checkCudaErrors( cudaGLRegisterBufferObject( positionVBO ) );
  checkCudaErrors( cudaGLRegisterBufferObject( velocityVBO ) );

  initCuda(nBoids);

  // Start GLUT Loop
  glutMainLoop();

  return 0;
}

__host__
void printDeviceProps() {
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

__host__
void Init(int argc, char* argv[]) {

  // Print out GPU Details
  printDeviceProps();

  // Create Window
  glutInit(&argc, argv);
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
  glutInitWindowSize(window_width, window_height);
  glutInitWindowPosition(0, 0);
  glutCreateWindow("cuda-boids");

  timeSinceLastFrame = glutGet(GLUT_ELAPSED_TIME);

  // Set GLUT functions for simulation
  glutReshapeFunc(windowResize);
  glutDisplayFunc(Render);
  glutIdleFunc(idleSim);
  glutKeyboardFunc(Keyboard);
  glutMotionFunc(mouseMotion);

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
  //gluPerspective(60.0, (double)window_width / (double)window_height, 0.1, 10.0);

  /*
  // Init GLEW
  glewInit();
  GLenum err = glewInit();
  if (GLEW_OK != err) {
      // Problem: glewInit failed, something is seriously wrong.
      std::cout << "glewInit failed, aborting." << std::endl;
      exit (1);
  }*/

  initVAO();
}

__host__
void initVAO(void) {
  GLfloat *bodies     = new GLfloat[4*(nBoids)];
  GLfloat *velocities = new GLfloat[3*(nBoids)];
  GLuint *bindices    = new GLuint [nBoids];

    for(int i = 0; i < nBoids; i++) {
      bodies[4*i+0] = 0.0f;
      bodies[4*i+1] = 0.0f;
      bodies[4*i+2] = 0.0f;
      bodies[4*i+3] = 1.0f;

      velocities[3*i+0] = 0.0f;
      velocities[3*i+1] = 0.0f;
      velocities[3*i+2] = 0.0f;

      bindices[i] = i;
    }

  glGenBuffers(1, &positionVBO);
  glGenBuffers(1, &velocityVBO);
  glGenBuffers(1, &IBO);

  glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
  glBufferData(GL_ARRAY_BUFFER, 4*(nBoids)*sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);
  glBufferData(GL_ARRAY_BUFFER, 3*(nBoids)*sizeof(GLfloat), velocities, GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (nBoids)*sizeof(GLuint), bindices, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  delete[] bodies;
  delete[] bindices;
  delete[] velocities;
}

// Idle function for GLUT, called when nothing is happening
__host__
void idleSim() {
  glutPostRedisplay();
}

// Called when OpenGL Window is resized to handle scaling
__host__
void windowResize(int height, int width) {
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (double)width / (double)height, 0.1, 10.0);
}

// Controls for simulation
__host__
void Keyboard(unsigned char key, int x, int y) {
  if (key == GLUT_KEY_RIGHT) {
    rY += 15;
  } else if (key == GLUT_KEY_LEFT) {
    rY -= 15;
  } else if (key == GLUT_KEY_DOWN) {
    rX -= 15;
  } else if (key == GLUT_KEY_UP) {
    rX += 15;
  } else if (key == 27) { // Escape
    exit(1);
  }

  // Request display update
  glutPostRedisplay();
}

// Function called when incorrect command line syntax is called or -h flag is passed
__host__
void help() {
   fprintf(stderr,"./boids --help|-h --nboids|-n \n");
}

int timebase = 0;
int frame = 0;

// Main render function for GLUT which loops until exit
__host__
void Render() {

  static float fps = 0;
  frame++;
  int time=glutGet(GLUT_ELAPSED_TIME);
  if (time - timebase > 1000) {
    fps = frame*1000.0f/(time-timebase);
    timebase = time;
    frame = 0;
  }
  float executionTime = glutGet(GLUT_ELAPSED_TIME) - timeSinceLastFrame;
  timeSinceLastFrame = glutGet(GLUT_ELAPSED_TIME);

  // Launch Kernel
  runCUDA();

  char title[100];
  sprintf( title, "cuda-boids [%d boids] [%0.2f fps] [%0.2fms] ", nBoids, fps, executionTime);
  glutSetWindowTitle(title);

  // Clear screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set View Matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Render from VBO
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
  glVertexAttribPointer((GLuint)0, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);
  glVertexAttribPointer((GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);

  glPointSize(4.0f);
  glDrawElements(GL_POINTS, nBoids, GL_UNSIGNED_INT, 0);

  glDisableVertexAttribArray(0);

  // Perspective modifications
  glRotatef(rX, 1.0, 0.0, 0.0);
  glRotatef(rY, 0.0, 1.0, 0.0);

  // Switch Buffers to show rendered
  glutPostRedisplay();
  glutSwapBuffers();

}

// Runs all CUDA related functions
__host__
void runCUDA() {

  float* dptrvert = NULL;
  float* velptr = NULL;
  checkCudaErrors( cudaGLMapBufferObject((void**)&dptrvert, positionVBO) );
  checkCudaErrors( cudaGLMapBufferObject((void**)&velptr, velocityVBO) );

  cudaFlockingUpdateWrapper(nBoids, 0.5, seekTarget);
  cudaUpdateVBO(nBoids, dptrvert, velptr);

  // unmap buffer object
  checkCudaErrors( cudaGLUnmapBufferObject(positionVBO) );
  checkCudaErrors( cudaGLUnmapBufferObject(velocityVBO) );
}

__host__
void mouseMotion(int x, int y) {
  float dx, dy;
  dx = (float)(x - mouse_old_x);
  dy = (float)(y - mouse_old_y);

  viewPhi   += 0.005f*dx;
  viewTheta += 0.005f*dy;
  seekTarget.x = 400.0f*sin(viewTheta)*sin(viewPhi);
  seekTarget.y = 400.0f*cos(viewTheta);
  seekTarget.z = 400.0f*sin(viewTheta)*cos(viewPhi);


  mouse_old_x = x;
  mouse_old_y = y;
}
