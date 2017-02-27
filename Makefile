COMMON	= ./common

DBG      ?=
NVCC     ?= nvcc
#CUDA_HOME?= $(TACC_CUDA_DIR)
CUDA_HOME?= /Developer/NVIDIA/CUDA-8.0/
NVFLAGS  = -I$(CUDA_HOME)include -I$(CUDA_HOME)samples/common/inc --ptxas-options="-v" -gencode=arch=compute_35,code=\"sm_35,compute_35\"
CXXFLAGS = -O3 -I. -I$(COMMON) $(DBG)

EXEC = main

all: $(EXEC)

OBJS = $(EXEC:=.o)
DEPS = $(OBJS:.o=.d)

-include $(DEPS)

# Load common make options
include $(COMMON)/Makefile.common
LDFLAGS	= $(COMMON_LIBS) -lcudart -Xlinker -framework,OpenGL,-framework,GLUT #-L$(CUDA_HOME)lib64 -lglut -lGLU -lGL

%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
#$(NVCC) -MM $(CXXFLAGS) $< > $*.d

kernel: kernel.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o kernel $^ $(LDFLAGS)

main: main.cpp kernel.o $(COMMON_OBJS)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o main $^ $(LDFLAGS)

clean: clean_common
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
