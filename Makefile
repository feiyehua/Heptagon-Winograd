CFLAG = -O4 -g 

BUILD_DIR = build

all:build/filter_transform_cuda.o build/image_transform_cuda.o build/driver.o build/winograd.o
	nvcc build/* -std=c++11 ${CFLAG} -o winograd

build/driver.o:
	gcc -c driver.cc -std=c++11 ${CFLAG} -o build/driver.o

build/winograd.o:
	nvcc -c winograd.cc -std=c++11 ${CFLAG} -o build/winograd.o

build/filter_transform_cuda.o:filter_transform.cu
	nvcc -c filter_transform.cu -std=c++11 ${CFLAG} -o build/filter_transform.o

build/image_transform_cuda.o:
	nvcc -c image_transform.cu -std=c++11 ${CFLAG} -o build/image_transform.o



clean:
	rm -f winograd
	rm -rf build