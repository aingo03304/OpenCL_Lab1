
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
