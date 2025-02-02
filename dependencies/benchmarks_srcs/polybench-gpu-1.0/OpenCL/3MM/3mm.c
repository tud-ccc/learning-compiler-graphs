#include <libcecl.h>
/**
 * 3mm.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size. */
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;


char str_temp[1024];

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem c_mem_obj;
cl_mem d_mem_obj;
cl_mem e_mem_obj;
cl_mem f_mem_obj;
cl_mem g_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NI; i++)
	{
		for (j=0; j < NL; j++)
		{
			if (percentDiff(G[i*NL + j], G_outputFromGpu[i*NL + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;				
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}

void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("3mm.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
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


void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, DATA_TYPE* G)
{
	a_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NI * NK, NULL, &errcode);
	b_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NK * NJ, NULL, &errcode);
	c_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NJ * NM, NULL, &errcode);
	d_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NM * NL, NULL, &errcode);
	e_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);
	f_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NJ * NL, NULL, &errcode);
	g_mem_obj = CECL_BUFFER(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NL, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = CECL_WRITE_BUFFER(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NK, A, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NK * NJ, B, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, c_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NJ * NM, C, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, d_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NM * NL, D, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, e_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ, E, 0, NULL, NULL);	
	errcode = CECL_WRITE_BUFFER(clCommandQue, f_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NJ * NL, F, 0, NULL, NULL);
	errcode = CECL_WRITE_BUFFER(clCommandQue, g_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NL, G, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
 }

void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = CECL_PROGRAM_WITH_SOURCE(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = CECL_PROGRAM(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernels
	clKernel1 = CECL_KERNEL(clProgram, "mm3_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clKernel2 = CECL_KERNEL(clProgram, "mm3_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clKernel3 = CECL_KERNEL(clProgram, "mm3_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel()
{
	double t_start, t_end;

	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;
	int nm = NM;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	t_start = rtclock();
	
	// Set the arguments of the kernel
	errcode =  CECL_SET_KERNEL_ARG(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel1, 2, sizeof(cl_mem), (void *)&e_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel1, 3, sizeof(int), (void *)&ni);
	errcode |= CECL_SET_KERNEL_ARG(clKernel1, 4, sizeof(int), (void *)&nj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel1, 5, sizeof(int), (void *)&nk);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	// Execute the OpenCL kernel

	errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);	
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clEnqueueBarrier(clCommandQue);

	globalWorkSize[0] = (size_t)ceil(((float)NL) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	errcode =  CECL_SET_KERNEL_ARG(clKernel2, 0, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel2, 1, sizeof(cl_mem), (void *)&d_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel2, 2, sizeof(cl_mem), (void *)&f_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel2, 3, sizeof(int), (void *)&nj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel2, 4, sizeof(int), (void *)&nl);
	errcode |= CECL_SET_KERNEL_ARG(clKernel2, 5, sizeof(int), (void *)&nm);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	// Execute the OpenCL kernel
	errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);	
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clEnqueueBarrier(clCommandQue);

	globalWorkSize[0] = (size_t)ceil(((float)NL) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	errcode =  CECL_SET_KERNEL_ARG(clKernel3, 0, sizeof(cl_mem), (void *)&e_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel3, 1, sizeof(cl_mem), (void *)&f_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel3, 2, sizeof(cl_mem), (void *)&g_mem_obj);
	errcode |= CECL_SET_KERNEL_ARG(clKernel3, 3, sizeof(int), (void *)&ni);
	errcode |= CECL_SET_KERNEL_ARG(clKernel3, 4, sizeof(int), (void *)&nl);
	errcode |= CECL_SET_KERNEL_ARG(clKernel3, 5, sizeof(int), (void *)&nj);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	// Execute the OpenCL kernel
	errcode = CECL_ND_RANGE_KERNEL(clCommandQue, clKernel3, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);	
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
	errcode = clReleaseKernel(clKernel3);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseMemObject(c_mem_obj);
	errcode = clReleaseMemObject(d_mem_obj);
	errcode = clReleaseMemObject(e_mem_obj);
	errcode = clReleaseMemObject(f_mem_obj);
	errcode = clReleaseMemObject(g_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
	int i,j,k;
	
	/* E := A*B */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			E[i*NJ + j] = 0;
			for (k = 0; k < NK; ++k)
			{
				E[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
			}
		}
	}
		
	/* F := C*D */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			F[i*NL + j] = 0;
			for (k = 0; k < NM; ++k)
			{
				F[i*NL + j] += C[i*NM + k] * D[k*NL + j];
			}
		}
	}

  	/* G := E*F */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			G[i*NL + j] = 0;
			for (k = 0; k < NJ; ++k)
			{
				G[i*NL + j] += E[i*NJ + k] * F[k*NL + j];
			}
		}
	}
}


int main(void) 
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;
	DATA_TYPE* G_outputFromGpu;

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(NJ*NM*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NM*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	F = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	G = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));
	G_outputFromGpu = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));

	int i;
	init_array(A, B, C, D);
	read_cl_file();
	cl_initialization();
	cl_mem_init(A, B, C, D, E, F, G);
	cl_load_prog();

	cl_launch_kernel();

	errcode = CECL_READ_BUFFER(clCommandQue, g_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NL, G_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

	t_start = rtclock();
	mm3_cpu(A, B, C, D, E, F, G);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(G, G_outputFromGpu);
	cl_clean_up();

	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);
	free(G_outputFromGpu);

	return 0;
}

