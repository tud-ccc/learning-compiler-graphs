#include <libcecl.h>
/**
 * mvt.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define N 4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256
#define DIM_LOCAL_WORK_GROUP_Y 1

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef double DATA_TYPE;

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem x1_mem_obj;
cl_mem x2_mem_obj;
cl_mem y1_mem_obj;
cl_mem y2_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;
const int LIST_SIZE = N;
char str_temp[1024];

void compareResults(DATA_TYPE* x1, DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2, DATA_TYPE* x2_outputFromGpu)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<N; i++) 
	{
		if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("mvt.cl", "r");
	if (!fp) {
		fprintf(stdout, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_arrays(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2)
{
    	int i, j;
	
	for (i=0; i<N; i++) 
   	{
        	x1[i] = 0.0;
        	x2[i] = 0.0;
		y_1[i] = 0.0;
        	y_2[i] = 0.0;
		
		for (j=0; j<N; j++)
		{
			a[i*N + j] = (DATA_TYPE)(i+j+1.0)/N;
		}
	}
}


void cl_initialization()
{	
	// Get platform and device information
	getOpenCLPlatformIdAndDeviceId(CL_DEVICE_TYPE_GPU, &platform_id, &device_id);

	// Create an OpenCL context
	clGPUContext = CECL_CREATE_CONTEXT( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = CECL_CREATE_COMMAND_QUEUE(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2)
{
	a_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * N, NULL, &errcode);
	x1_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	x2_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	y1_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	y2_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = CECL_WRITE_BUFFER(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, a, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, x1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x1, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, x2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x2, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, y1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y_1, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, y2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y_2, 0, NULL, NULL);

	if(errcode != CL_SUCCESS) printf("Error in writing buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = CECL_PROGRAM_WITH_SOURCE(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = CECL_PROGRAM(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program %d\n",errcode);
		
	// Create the 1st OpenCL kernel
	clKernel1 = CECL_KERNEL(clProgram, "mvt_kernel1", &errcode);
	// Create the 2nd OpenCL kernel
	clKernel2 = CECL_KERNEL(clProgram, "mvt_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	double t_start, t_end;

	int n = N;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;

	t_start = rtclock();
	
	// Set the arguments of the kernel
	errcode =  CECL_SET_KERNEL_ARG(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel1, 1, sizeof(cl_mem), (void *)&x1_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel1, 2, sizeof(cl_mem), (void *)&y1_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel1, 3, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
 
	// Execute the OpenCL kernel
	errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");

	// Set the arguments of the kernel
	errcode =  CECL_SET_KERNEL_ARG(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel2, 1, sizeof(cl_mem), (void *)&x2_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel2, 2, sizeof(cl_mem), (void *)&y2_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel2, 3, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");


	clFinish(clCommandQue);

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(x1_mem_obj);
	errcode = clReleaseMemObject(x2_mem_obj);
	errcode = clReleaseMemObject(y1_mem_obj);
	errcode = clReleaseMemObject(y2_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
	int i, j, k, l;
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
       			x1[i] = x1[i] + a[i*N + j] * y1[j];
        	}
    	}
	
	for (k=0; k<N; k++) 
	{
		for (l=0; l<N; l++) 
		{
 		       	x2[k] = x2[k] + a[k*N + l] * y2[l];
      		}
    	}
}


int main(void) 
{
	double t_start, t_end;
	
	DATA_TYPE* a;
	DATA_TYPE* x1;
	DATA_TYPE* x2;
	DATA_TYPE* x1_outputFromGpu;
	DATA_TYPE* x2_outputFromGpu;
	DATA_TYPE* y_1;
	DATA_TYPE* y_2;

	a = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	x1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	x2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	x1_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	x2_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

	init_arrays(a, x1, x2, y_1, y_2);
	read_cl_file();
	cl_initialization();
	cl_mem_init(a, x1, x2, y_1, y_2);
	cl_load_prog();

	cl_launch_kernel();

	errcode = CECL_READ_BUFFER(clCommandQue, x1_mem_obj, CL_TRUE, 0, N*sizeof(DATA_TYPE), x1_outputFromGpu, 0, NULL, NULL);
	errcode = CECL_READ_BUFFER(clCommandQue, x2_mem_obj, CL_TRUE, 0, N*sizeof(DATA_TYPE), x2_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");   

	t_start = rtclock();
	runMvt(a, x1, x2, y_1, y_2);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);
	cl_clean_up();

	free(a);
	free(x1);
	free(x2);
	free(x1_outputFromGpu);
	free(x2_outputFromGpu);
	free(y_1);
	free(y_2);

	return 0;
}

