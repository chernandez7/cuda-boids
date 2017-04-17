
#include "main.h"
#include "kernel.h"

__host__
int main(int argc, char* argv[]) {
	/*
	// Parse command line parameters
	for (int i = 1; i < argc; ++i) {
#define check_index(i,str) \
  if ((i) >= argc) \
     { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			help();
			return 1;
		}
		else if (strcmp(argv[i], "--nboids") == 0 || strcmp(argv[i], "-n") == 0) {
			check_index(i + 1, "--nboids|-n");
			i++;
			if (isdigit(*argv[i]))
				nBoids = atoi(argv[i]);
		}
		else {
			fprintf(stderr, "Unknown option %s\n", argv[i]);
			help();
			return 1;
		}
	}
	*/
	nBoids = 100;

	// OpenGL / GLUT Initialization
	Init(argc, argv);

	cudaGLSetGLDevice(0);

	// Register VBO
	cudaGLRegisterBufferObject(positionVBO);
	cudaGLRegisterBufferObject(velocityVBO);
	cudaGLRegisterBufferObject(accelerationVBO);

	// CUDA Init
	initCuda(nBoids);

	// Perspective
	projection = glm::perspective(fovy, float(window_width) / float(window_height), zNear, zFar);
	view = glm::lookAt(cameraPosition, glm::vec3(0.0, 0.0, 0), glm::vec3(0, 1, 0));
	projection = projection * view;

	initShaders(program);

	glEnable(GL_DEPTH_TEST);

	glutReshapeFunc(windowResize);
	glutDisplayFunc(Render);
	glutKeyboardFunc(Keyboard);
	glutPassiveMotionFunc(mouseMotion);

	// Start GLUT Loop
	glutMainLoop();

	return 0;
}

__host__
void printDeviceProps() {
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, 0);
		fprintf(stderr, "   name:                           %s\n", props.name);
		fprintf(stderr, "   major.minor:                    %d.%d\n", props.major, props.minor);
		fprintf(stderr, "   totalGlobalMem:                 %lu (MB)\n", props.totalGlobalMem / (1024 * 1024));
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
	fprintf(stdout, "   Initalizing application.\n");
	// Print out GPU Details
	printDeviceProps();

	// Create Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(30, 30);
	glutCreateWindow("cuda-boids");

	// Init GLFW
	if (!glfwInit()) {
		std::cout << "glfw init failed" << std::endl;
	}

	// Init GLEW
	glewInit();
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		// Problem: glewInit failed, something is seriously wrong.
		std::cout << "glewInit failed, aborting." << std::endl;
		exit(1);
	}
	fprintf(stdout, "   Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	timeSinceLastFrame = glutGet(GLUT_ELAPSED_TIME);

	// For mouse location
	seekTarget = make_float3(0.0, 0.0, 0.0);

	initVAO();
}

__host__
void initVAO(void) {
	fprintf(stdout, "   Creating Vertex Array Objects.\n");

	GLfloat *bodies = new GLfloat[4 * (nBoids)];
	GLfloat *velocities = new GLfloat[3 * (nBoids)];
	GLfloat *accelerations = new GLfloat[3 * (nBoids)];
	GLuint *bindices = new GLuint[nBoids];

	// Initializing all positions and vel's at 0
	for (int i = 0; i < nBoids; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;

		velocities[3 * i + 0] = 0.0f;
		velocities[3 * i + 1] = 0.0f;
		velocities[3 * i + 2] = 0.0f;

		accelerations[3 * i + 0] = 0.0f;
		accelerations[3 * i + 1] = 0.0f;
		accelerations[3 * i + 2] = 0.0f;

		bindices[i] = i;
	}

	// Generate buffers for VBO
	glGenBuffers(1, &positionVBO);
	glGenBuffers(1, &velocityVBO);
	glGenBuffers(1, &accelerationVBO);
	glGenBuffers(1, &IBO);

	fprintf(stdout, "   Binding VBO to GL buffers.\n");
	// Assign VBO to buffer
	glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
	// Add bodies and size to buffer
	glBufferData(GL_ARRAY_BUFFER, 4 * (nBoids) * sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);

	// Same for vel
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);
	glBufferData(GL_ARRAY_BUFFER, 3 * (nBoids) * sizeof(GLfloat), velocities, GL_DYNAMIC_DRAW);

	// Same for acc
	glBindBuffer(GL_ARRAY_BUFFER, accelerationVBO);
	glBufferData(GL_ARRAY_BUFFER, 3 * (nBoids) * sizeof(GLfloat), accelerations, GL_DYNAMIC_DRAW);

	// Same for IBO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (nBoids) * sizeof(GLuint), bindices, GL_STATIC_DRAW);

	// Unbind buffers
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// No need for these anymore since in buffers
	delete[] bodies;
	delete[] bindices;
	delete[] velocities;
	delete[] accelerations;
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
		fprintf(stdout, "   Moving Y-axis up.\n");
		rY += 15;
	}
	else if (key == GLUT_KEY_LEFT) {
		fprintf(stdout, "   Moving Y-axis down.\n");
		rY -= 15;
	}
	else if (key == GLUT_KEY_DOWN) {
		fprintf(stdout, "   Moving X-axis left.\n");
		rX -= 15;
	}
	else if (key == GLUT_KEY_UP) {
		fprintf(stdout, "   Moving X-axis right.\n");
		rX += 15;
	}
	else if (key == 27) { // Escape
		fprintf(stdout, "   Exiting application.\n");
		exit(1);
	}

	// Request display update
	glutPostRedisplay();
}

// Function called when incorrect command line syntax is called or -h flag is passed
__host__
void help() {
	fprintf(stderr, "./boids --help|-h --nboids|-n \n");
}

int timebase = 0;
int frame = 0;

// Main render function for GLUT which loops until exit
__host__
void Render() {
	//fprintf(stdout, "   Entering Render Loop.\n");

	static float fps = 0;
	frame++;
	int time = glutGet(GLUT_ELAPSED_TIME);
	if (time - timebase > 1000) {
		fps = frame*1000.0f / (time - timebase);
		timebase = time;
		frame = 0;
	}
	float executionTime = glutGet(GLUT_ELAPSED_TIME) - timeSinceLastFrame;
	timeSinceLastFrame = glutGet(GLUT_ELAPSED_TIME);

	// Launch Kernel
	runCUDA();

	char title[100];
	sprintf(title, "[%d boids] [%0.2f fps] [%0.2f, %0.2f, %0.2f mouse] [%0.2fms]", nBoids, fps, seekTarget.x, seekTarget.y, seekTarget.z, executionTime);
	glutSetWindowTitle(title);

	// Clear screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set View Matrix
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();

	glUseProgram(program[PASS_THROUGH]);

	//fprintf(stdout, "   Drawing Vertices.\n");
	// Render from VBO
	glEnableVertexAttribArray(positionLocation);
	glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
	glVertexAttribPointer((GLuint)0, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glEnableVertexAttribArray(velocityLocation);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);
	glVertexAttribPointer((GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glEnableVertexAttribArray(accelerationLocation);
	glBindBuffer(GL_ARRAY_BUFFER, accelerationVBO);
	glVertexAttribPointer((GLuint)2, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);

	glPointSize(10.0f);
	glDrawElements(GL_POINTS, nBoids, GL_UNSIGNED_INT, 0);

	glDisableVertexAttribArray(0);

	// Perspective modifications
	//glRotatef(rX, 1.0, 0.0, 0.0);
	//glRotatef(rY, 0.0, 1.0, 0.0);

	// Switch Buffers to show rendered
	glutPostRedisplay();
	glutSwapBuffers();

}

// Runs all CUDA related functions
__host__
void runCUDA() {

	float* dptrvert = NULL;
	float* velptr = NULL;
	float* accptr = NULL;
	//fprintf(stdout, "   Mapping GL buffer.\n");
	cudaGLMapBufferObject((void**)&dptrvert, positionVBO);
	cudaGLMapBufferObject((void**)&velptr, velocityVBO);
	cudaGLMapBufferObject((void**)&accptr, accelerationVBO);

	flock(nBoids, window_width, window_height, seekTarget);
	cudaUpdateVBO(nBoids, dptrvert, velptr, accptr);

	// unmap buffer object
	cudaGLUnmapBufferObject(positionVBO);
	cudaGLUnmapBufferObject(velocityVBO);
	cudaGLUnmapBufferObject(accelerationVBO);

}

void initShaders(GLuint* program) {
	GLint location;

	program[1] = glslUtility::createProgram("shaders/planetVS.glsl", "shaders/planetGS.glsl", "shaders/planetFS.glsl", attributeLocations, 1);
	glUseProgram(program[1]);

	if ((location = glGetUniformLocation(program[1], "u_projMatrix")) != -1)
	{
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[1], "u_cameraPos")) != -1)
	{
		glUniform3fv(location, 1, &cameraPosition[0]);
	}
}

__host__
void mouseMotion(int x, int y) {
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	viewPhi += 0.005f*dx;
	viewTheta += 0.005f*dy;
	seekTarget.x = 400.0f*sin(viewTheta)*sin(viewPhi);
	seekTarget.y = 400.0f*cos(viewTheta);
	seekTarget.z = 400.0f*sin(viewTheta)*cos(viewPhi);

	mouse_old_x = x;
	mouse_old_y = y;
}
