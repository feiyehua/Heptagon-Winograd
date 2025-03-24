CFLAG = -O4 -g -fopenmp -I/usr/local/cuda/include/ -lcudart -L/usr/local/cuda/lib64

NVCCFLAG = -O4 -g

BUILD_DIR = build

all:build/filter_transform_cuda.o build/image_transform_cuda.o build/output_transform_cuda.o build/driver.o build/winograd.o
	nvcc build/* -std=c++11 ${NVCCFLAG} -Xcompiler -fopenmp -o winograd

build/driver.o:driver.cc
	g++ -c driver.cc -std=c++11 ${CFLAG} -o build/driver.o

build/winograd.o:winograd.cc
	g++ -c winograd.cc -std=c++11 ${CFLAG} -o build/winograd.o

build/filter_transform_cuda.o:filter_transform.cu
	nvcc -c filter_transform.cu -std=c++11 ${NVCCFLAG} -o build/filter_transform.o

build/image_transform_cuda.o:image_transform.cu
	nvcc -c image_transform.cu -std=c++11 ${NVCCFLAG} -o build/image_transform.o

build/output_transform_cuda.o:output_transform.cu
	nvcc -c output_transform.cu -std=c++11 ${NVCCFLAG} -o build/output_transform.o

clean:
	rm -f winograd
	rm -rf build