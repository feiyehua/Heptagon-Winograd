# Compiler flags
CFLAGS = -Ofast -g -std=c++11 -fopenmp -I./include -I/usr/local/cuda/include
NVCCFLAGS = -O4 -g -rdc=true -Xptxas -v -std=c++11 -I./include

# Directories
SRC_CUDA = src/cuda
SRC_CPU = src/cpu
INCLUDE_DIR = include
BUILD_DIR = build

CUDA_SOURCES := $(wildcard src/cuda/*.cu)
CPU_SOURCES := $(wildcard src/cpu/*.cc)

# Object files
# CUDA_OBJECTS := $(patsubst %.cu, $(BUILD_DIR)/%.o, $(CUDA_SOURCES))
# CPU_OBJECTS := $(patsubst %.cc, $(BUILD_DIR)/%.o, $(CPU_SOURCES))
CPU_OBJECTS := $(patsubst src/cpu/%.cc, $(BUILD_DIR)/%.o, $(CPU_SOURCES))
CUDA_OBJECTS := $(patsubst src/cuda/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SOURCES))

# CUDA_OBJECTS := $(addprefix $(BUILD_DIR)/, filter_transform.o image_transform.o output_transform.o cublas_sgemm.o)
# CPU_OBJECTS := $(addprefix $(BUILD_DIR)/, driver.o winograd.o)

# Targets
all: winograd

winograd: $(CUDA_OBJECTS) $(CPU_OBJECTS) $(BUILD_DIR)/driver.o | make_build_dir
	nvcc -o $@ $^ -Xptxas -v -Xcompiler -fopenmp -lcublas

$(BUILD_DIR)/%.o: $(SRC_CUDA)/%.cu | make_build_dir
	nvcc -c $< $(NVCCFLAGS) -o $@

$(BUILD_DIR)/%.o: $(SRC_CPU)/%.cc | make_build_dir
	g++ -c $< $(CFLAGS) -o $@

$(BUILD_DIR)/driver.o:driver.cc | make_build_dir
	g++ -c driver.cc $(CFLAGS) -o $(BUILD_DIR)/driver.o

make_build_dir:
	mkdir -p build

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR) winograd
