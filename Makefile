COMMON	= ./common

DBG      ?=
NVCC     ?= nvcc
CUDA_HOME?= /usr/local/cuda
NVFLAGS  = -I$(CUDA_HOME)/include --ptxas-options="-v" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -Xcompiler
CXXFLAGS = -O3 -I. -I$(COMMON) $(DBG)

EXEC = cuda-boids

all: $(EXEC)

OBJS = $(EXEC:=.o)
DEPS = $(OBJS:.o=.d)

-include $(DEPS)

# Load common make options
include $(COMMON)/Makefile.common
LDFLAGS	= $(COMMON_LIBS) -lcudart -L$(CUDA_HOME)/lib64

%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
#$(NVCC) -MM $(CXXFLAGS) $< > $*.d

vector3f: vector3f.o
	$(NVCC) $(CXXFLAGS) -o vector3f $^ $(LDFLAGS)

flock: flock.o
	$(NVCC) $(CXXFLAGS) -o flock $^ $(LDFLAGS)

boids: boids.o flock.o vector3f.o
	$(NVCC) $(CXXFLAGS) -o boids $^ $(LDFLAGS)

cuda-boids: cuda-boids.o boids.o flock.o vector3f.o $(COMMON_OBJS)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o cuda-boids $^ $(LDFLAGS)

clean: clean_common
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
