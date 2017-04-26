
#include "main.h"
#include "kernel.h"

/*****************************************************************
*
*	Main
*
****************************************************************/

__host__
int main(int argc, char* argv[]) {
	
	// Parse command line parameters
	for (int i = 1; i < argc; i++) {
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
		else if (strcmp(argv[i], "--mouse") == 0 || strcmp(argv[i], "-m") == 0) {
		followMouse = true;
		} else if (strcmp(argv[i], "--naive") == 0) {
		naive = true;
		} else {
			fprintf(stderr, "Unknown option %s\n", argv[i]);
			help();
			return 1;
		}
	}
	
	//nBoids = 512;
	//followMouse = false;
	//naive = false;

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
	projection = glm::perspective(
		fovy,										// Field of View
		float(window_width) / float(window_height),	// Aspect Ratio
		zNear,										// Near Plane
		zFar										// Far Plane
	);
	view = glm::lookAt(cameraPosition, glm::vec3(0.0, 0.0, 0), glm::vec3(0, 1, 0));
	projection = projection * view;

	initShaders(program);

	printControls();

	glEnable(GL_DEPTH_TEST);

	glutReshapeFunc(windowResize);
	glutDisplayFunc(Render);
	glutKeyboardFunc(Keyboard);
	glutPassiveMotionFunc(mouseMotion);

	// Start GLUT Loop
	glutMainLoop();

	return 0;
}

/*****************************************************************
*
*	OpenGL Init
*
****************************************************************/

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
		fprintf(stderr, "   maxThreadsPerBlock:             %d\n\n", props.maxThreadsPerBlock);
	}
}

__host__
void printControls() {
	fprintf(stderr, "   **************** Controls ****************\n");
	fprintf(stderr, "   W: Increase Separation Radius\n");
	fprintf(stderr, "   T: Increase Alignment Radius\n");
	fprintf(stderr, "   I: Increase Cohesion Radius\n\n");

	fprintf(stderr, "   S: Decrease Separation Radius\n");
	fprintf(stderr, "   G: Decrease Alignment Radius\n");
	fprintf(stderr, "   K: Decrease Cohesion Radius\n\n");

	fprintf(stderr, "   A: Increase Separation Weight\n");
	fprintf(stderr, "   F: Increase Alignment Weight\n");
	fprintf(stderr, "   J: Increase Cohesion Weight\n\n");

	fprintf(stderr, "   D: Decrease Separation Weight\n");
	fprintf(stderr, "   J: Decrease Alignment Weight\n");
	fprintf(stderr, "   L: Decrease Cohesion Weight\n\n");

	fprintf(stderr, "   R: Reset Parameter Values to Default\n");

	fprintf(stderr, "   *****************************************\n\n");
}

__host__
void Init(int argc, char* argv[]) {
	fprintf(stdout, "   Initalizing application.\n\n");
	// Print out GPU Details
	printDeviceProps();

	// Create Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(0, 30);
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

	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

	initVAO();
}

void initShaders(GLuint* program) {
	GLint location;

	program[1] = glslUtility::createProgram(
		"shaders/particleVS.glsl",
		"shaders/particleGS.glsl",
		"shaders/particleFS.glsl",
		attributeLocations,
		1
	);
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
void initVAO(void) {
	fprintf(stdout, "   Creating Vertex Array Objects.\n");

	GLfloat* bodies = new GLfloat[4 * (nBoids)];
	GLfloat* velocities = new GLfloat[3 * (nBoids)];
	GLfloat* accelerations = new GLfloat[3 * (nBoids)];
	GLuint* bindices = new GLuint[nBoids];

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

__host__
void help() {
	fprintf(stderr, "./boids --help|-h --nboids|-n --mouse|-m --naive \n");
}

/*****************************************************************
*
*	CUDA Wrapper
*
****************************************************************/

// Runs all CUDA related functions
__host__
void runCUDA(bool followMouse, float sep_dist, float sep_weight, float ali_dist, float ali_weight, float coh_dist, float coh_weight) {

	float* dptrvert = NULL;
	float* velptr = NULL;
	float* accptr = NULL;

	// Map BO to array
	cudaGLMapBufferObject((void**)&dptrvert, positionVBO);
	cudaGLMapBufferObject((void**)&velptr, velocityVBO);
	cudaGLMapBufferObject((void**)&accptr, accelerationVBO);

	// Call wrappers for kernels
	flock(nBoids, window_width, window_height, seekTarget, followMouse, naive, sep_dist, sep_weight, ali_dist, ali_weight, coh_dist, coh_weight);
	cudaUpdateVBO(nBoids, dptrvert, velptr, accptr);

	// Unmap buffer object
	cudaGLUnmapBufferObject(positionVBO);
	cudaGLUnmapBufferObject(velocityVBO);
	cudaGLUnmapBufferObject(accelerationVBO);

}

/*****************************************************************
*
*	GLUT Functions
*
****************************************************************/

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

__host__
void windowResize(int height, int width) {
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
}

__host__
void Keyboard(unsigned char key, int x, int y) {
	if (key == 27) { // Escape
		fprintf(stdout, "   Exiting application.\n");
		exit(1);
	}
	else if (key == 'w') { // sep up
		if (sep_dist <= 0)	sep_dist = 0;
		else				sep_dist += 10;
	}
	else if (key == 'a') { // sep weight down
		if (sep_weight <= 0)	sep_weight = 0;
		else					sep_weight -= 0.1;
	}
	else if (key == 's') { // sep down
		if (sep_dist <= 0)	sep_dist = 0;
		else				sep_dist -= 10;
	}
	else if (key == 'd') { // sep weight up
		if (sep_weight <= 0)	sep_weight = 0;
		else					sep_weight += 0.1;
	}
	else if (key == 't') { // ali up
		if (ali_dist <= 0)	ali_dist = 0;
		else				ali_dist += 10;
	}
	else if (key == 'f') { // ali weight down
		if (ali_weight <= 0)	ali_weight = 0;
		else					ali_weight -= 0.1;
	}
	else if (key == 'g') { // ali down
		if (ali_dist <= 0)	ali_dist = 0;
		else				ali_dist -= 10;
	}
	else if (key == 'h') { // ali weight up
		if (ali_weight <= 0)	ali_weight = 0;
		else					ali_weight += 0.1;
	}
	else if (key == 'i') { // coh up
		if (coh_dist <= 0)	coh_dist = 0;
		else				coh_dist += 10;
	}
	else if (key == 'j') { // coh weight down
		if (coh_weight <= 0)	coh_weight = 0;
		else					coh_weight -= 0.1;
	}
	else if (key == 'k') { // coh down
		if (coh_dist <= 0)	coh_dist = 0;
		else				coh_dist -= 10;
	}
	else if (key == 'l') { // coh weight up
		if (coh_weight <= 0)	coh_weight = 0;
		else					coh_weight += 0.1;
	}
	else if (key == 'r') { // reset values
		sep_dist = 100;
		ali_dist = 400;
		coh_dist = 300;

		sep_weight = 1.0f;
		ali_weight = 1.5f;
		coh_weight = 1.0f;
	}

	// Request display update
	glutPostRedisplay();
}

__host__
void Render() {

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
	runCUDA(followMouse, sep_dist, sep_weight, ali_dist, ali_weight, coh_dist, coh_weight);

	char title[200];
	if (followMouse) {
		// sep_dist not taken into consideration when mouse is target
		sprintf(title, "[%d boids] [%0.2f fps] [%0.2f, %0.2f, %0.2f mouse] dist: (%0.2f ali, %0.2f coh) weight: (%0.2f sep, %0.2f ali, %0.2f coh)",
			nBoids, fps, seekTarget.x, seekTarget.y, seekTarget.z, ali_dist, coh_dist, sep_weight, ali_weight, coh_weight);
	} else {
		sprintf(title, "[%d boids] [%0.2f fps] dist: (%0.2f sep, %0.2f ali, %0.2f coh) weight: (%0.2f sep, %0.2f ali, %0.2f coh)",
			nBoids, fps, sep_dist, ali_dist, coh_dist, sep_weight, ali_weight, coh_weight);
	}
	glutSetWindowTitle(title);

	// Clear screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(program[PASS_THROUGH]);

	// Render from VBO
	glEnableVertexAttribArray(positionLocation);
	glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glEnableVertexAttribArray(velocityLocation);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);
	glVertexAttribPointer((GLuint)velocityLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glEnableVertexAttribArray(accelerationLocation);
	glBindBuffer(GL_ARRAY_BUFFER, accelerationVBO);
	glVertexAttribPointer((GLuint)accelerationLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);

	glPointSize(4.0f);
	glDrawElements(GL_POINTS, nBoids, GL_UNSIGNED_INT, 0); // where draw happens

	glDisableVertexAttribArray(positionLocation);

	// Switch Buffers to show rendered
	glutPostRedisplay();
	glutSwapBuffers();

}
