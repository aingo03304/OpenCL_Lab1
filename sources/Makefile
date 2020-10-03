
NVCC=nvcc
INCLUDE= -I../libgputk
LIBgpuTK= -L../build-dir -lgputk -lcuda
LIBS= -lOpenCL $(LIBgpuTK)

all: template


template:
	$(NVCC) -std=c++11 template.cpp $(INCLUDE) $(LIBS) -o OpenCLVectorAdd_Template 


dataset_generator:
	$(NVCC) -std=c++11 dataset_generator.cpp $(INCLUDE) $(LIBS) -o dataset_generator


clean:
	rm -f OpenCLVectorAdd_Template dataset_generator

test:
	$(NVCC) -std=c++11 template.cpp $(INCLUDE) $(LIBS) -o OpenCLVectorAdd_Template	
	for n in 0 1 2 3 4 5 6 7 8 9 ; \
	do \
		./OpenCLVectorAdd_Template -e OpenCLVectorAdd/Dataset/$$n/output.raw \
	       		-i OpenCLVectorAdd/Dataset/$$n/input0.raw,OpenCLVectorAdd/Dataset/$$n/input1.raw \
			-o OpenCLVectorAdd/Dataset/$$n/infer.raw -t vector ; \
	done
