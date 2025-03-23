CFLAG = -O4 -g

BUILD_DIR = build

all:build/filter_transform_cuda.o build/image_transform_cuda.o build/driver.o build/winograd.o
	nvcc build/* -G -std=c++11 ${CFLAG} -o winograd

build/driver.o:driver.cc
	gcc -c driver.cc -std=c++11 ${CFLAG} -o build/driver.o

build/winograd.o:winograd.cc
	nvcc -c winograd.cc -G -std=c++11 ${CFLAG} -o build/winograd.o

build/filter_transform_cuda.o:filter_transform.cu
	nvcc -c filter_transform.cu -G -std=c++11 ${CFLAG} -o build/filter_transform.o

build/image_transform_cuda.o:image_transform.cu
	nvcc -c image_transform.cu -G -std=c++11 ${CFLAG} -o build/image_transform.o



clean:
	rm -f winograd
	rm -rf build