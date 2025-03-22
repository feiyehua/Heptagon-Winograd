CFLAG = -O3 -g 

all:
	nvcc driver.cc winograd.cc filter_transform.cu -std=c++11 ${CFLAG} -o winograd

clean:
	rm -f winograd