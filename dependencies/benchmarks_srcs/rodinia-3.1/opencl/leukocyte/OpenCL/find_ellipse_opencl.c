#include <libcecl.h>
#include "find_ellipse.h"
#include "find_ellipse_opencl.h"
#include "OpenCL_helper_library.h"
#include <CL/cl.h>
#include <stdio.h>


// Defined if we want to use images/textures
/* #define USE_IMAGE */


// The number of sample points in each ellipse (stencil)
#define NPOINTS 150
// The maximum radius of a sample ellipse
#define MAX_RAD 20
// The total number of sample ellipses
#define NCIRCLES 7
// The size of the structuring element used in dilation
#define STREL_SIZE (12 * 2 + 1)


// Matrix used to store the maximal GICOV score at each pixels
// Produced by the GICOV kernel and consumed by the dilation kernel
cl_mem device_gicov;

// Device arrays holding the stencil parameters used by the GICOV kernel
cl_mem c_sin_angle, c_cos_angle, c_tX, c_tY;

// Dilate kernel
cl_kernel dilate_kernel;


// Sets up and invokes the GICOV kernel and returns its output
float *GICOV_OpenCL(int grad_m, int grad_n, float *host_grad_x, float *host_grad_y) {
	
	cl_int error;

	int MaxR = MAX_RAD + 2;

	// Allocate device buffers and transfer data
	unsigned int grad_mem_size = sizeof(float) * grad_m * grad_n;
	cl_mem device_grad_x, device_grad_y;
	#ifdef USE_IMAGE
	// Define the image parameters
	cl_image_format image_format;
	image_format.image_channel_order = CL_R;
	image_format.image_channel_data_type = CL_FLOAT;
	// Create images (textures)
	device_grad_x = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &image_format, grad_m, grad_n, 0, host_grad_x, &error);
	check_error(error, __FILE__, __LINE__);
	device_grad_y = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &image_format, grad_m, grad_n, 0, host_grad_y, &error);
	check_error(error, __FILE__, __LINE__);
	#else
	// Create buffers
	device_grad_x = CECL_BUFFER(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, grad_mem_size, host_grad_x, &error);
	check_error(error, __FILE__, __LINE__);
	device_grad_y = CECL_BUFFER(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, grad_mem_size, host_grad_y, &error);
	check_error(error, __FILE__, __LINE__);
	#endif

	// Allocate & initialize device memory for result
	// (some elements are not assigned values in the kernel)
	// Since there is no OpenCL version of cudaMemset, we first allocate and initialize
	//  the host-side copy of the buffer and then transfer that to the device
	float *host_gicov = (float *) malloc(grad_mem_size);
	memset(host_gicov, 0, grad_mem_size);
	device_gicov = CECL_BUFFER(context, CL_MEM_READ_WRITE | /*CL_MEM_COPY_HOST_PTR*/ CL_MEM_USE_HOST_PTR, grad_mem_size, host_gicov, &error);
	check_error(error, __FILE__, __LINE__);
	
	// Load the kernel source from the file
	const char *source = load_kernel_source("find_ellipse_kernel.cl");
	size_t sourceSize = strlen(source);
	
	// Compile the kernel
	cl_program program = CECL_PROGRAM_WITH_SOURCE(context, 1, &source, &sourceSize, &error);
    check_error(error, __FILE__, __LINE__);
	#ifdef USE_IMAGE
	error = CECL_PROGRAM(program, 1, &device, "-D USE_IMAGE", NULL, NULL);
	#else
	error = CECL_PROGRAM(program, 1, &device, NULL, NULL, NULL);
	#endif
	// Show compiler warnings/errors
	static char log[65536]; memset(log, 0, sizeof(log));
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
	if (strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    check_error(error, __FILE__, __LINE__);

	// Create both kernels (GICOV and dilate)
    cl_kernel GICOV_kernel = CECL_KERNEL(program, "GICOV_kernel", &error);
    check_error(error, __FILE__, __LINE__);
	dilate_kernel = CECL_KERNEL(program, "dilate_kernel", &error);
    check_error(error, __FILE__, __LINE__);

	
    // Setup execution parameters
    cl_int local_work_size = grad_m - (2 * MaxR); 
    cl_int num_work_groups = grad_n - (2 * MaxR);
   
	// Set the kernel arguments
	CECL_SET_KERNEL_ARG(GICOV_kernel, 0, sizeof(cl_int), (void *) &grad_m);
	CECL_SET_KERNEL_ARG(GICOV_kernel, 1, sizeof(cl_mem), (void *) &device_grad_x);
	CECL_SET_KERNEL_ARG(GICOV_kernel, 2, sizeof(cl_mem), (void *) &device_grad_y);
	CECL_SET_KERNEL_ARG(GICOV_kernel, 3, sizeof(cl_mem), (void *) &c_sin_angle);
	CECL_SET_KERNEL_ARG(GICOV_kernel, 4, sizeof(cl_mem), (void *) &c_cos_angle);
	CECL_SET_KERNEL_ARG(GICOV_kernel, 5, sizeof(cl_mem), (void *) &c_tX);
	CECL_SET_KERNEL_ARG(GICOV_kernel, 6, sizeof(cl_mem), (void *) &c_tY);
	CECL_SET_KERNEL_ARG(GICOV_kernel, 7, sizeof(cl_mem), (void *) &device_gicov);
	CECL_SET_KERNEL_ARG(GICOV_kernel, 8, sizeof(cl_int), (void *) &local_work_size);
	CECL_SET_KERNEL_ARG(GICOV_kernel, 9, sizeof(cl_int), (void *) &num_work_groups);

	size_t work_group_size = 256;
	size_t global_work_size = num_work_groups * local_work_size;
	if(global_work_size % work_group_size > 0)
	  global_work_size=(global_work_size / work_group_size+1)*work_group_size;

	printf("Find: local_work_size = %d, global_work_size = %d \n"
	       ,work_group_size, global_work_size);

	// Execute the GICOV kernel
	error = CECL_ND_RANGE_KERNEL(command_queue, GICOV_kernel, 1, NULL, &global_work_size, &work_group_size, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
	

	// Check for kernel errors
	error = clFinish(command_queue);
	check_error(error, __FILE__, __LINE__);

	// Copy the result to the host
	host_gicov = (cl_float *) CECL_MAP_BUFFER(command_queue, device_gicov, CL_TRUE, CL_MAP_READ, 0, grad_mem_size, 0, NULL, NULL, &error);
	check_error(error, __FILE__, __LINE__);

	// Cleanup memory
	clReleaseMemObject(device_grad_x);
	clReleaseMemObject(device_grad_y);

	return host_gicov;
}


// Constant device array holding the structuring element used by the dilation kernel
cl_mem c_strel;


// Sets up and invokes the dilation kernel and returns its output
float *dilate_OpenCL(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n) {

	cl_int error;
	
	// Allocate device memory for result
	unsigned int max_gicov_mem_size = sizeof(float) * max_gicov_m * max_gicov_n;
	cl_mem device_img_dilated = CECL_BUFFER(context, CL_MEM_WRITE_ONLY, max_gicov_mem_size, NULL, &error);
	check_error(error, __FILE__, __LINE__);
	
	#ifdef USE_IMAGE
	// Copy the input matrix of GICOV values to an image
	// Define the image parameters
	cl_image_format image_format;
	image_format.image_channel_order = CL_R;
	image_format.image_channel_data_type = CL_FLOAT;
	// Create the image
	cl_mem device_gicov_image = clCreateImage2D(context, CL_MEM_READ_ONLY, &image_format, max_gicov_m, max_gicov_n, 0, NULL, &error);
	check_error(error, __FILE__, __LINE__);
	// Copy the GICOV data to the image
	size_t offset[3] = {0, 0, 0};
	size_t region[3] = {max_gicov_m, max_gicov_n, 1};
	error = clEnqueueCopyBufferToImage(command_queue, device_gicov, device_gicov_image, 0, offset, region, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
	#endif
    
	// Setup execution parameters
	size_t global_work_size = max_gicov_m * max_gicov_n;
	size_t local_work_size = 176;
	// Make sure the global work size is a multiple of the local work size
	if (global_work_size % local_work_size != 0) {
		global_work_size = ((global_work_size / local_work_size) + 1) * local_work_size;
	}
	
	// Set the kernel arguments
	CECL_SET_KERNEL_ARG(dilate_kernel, 0, sizeof(cl_int), (void *) &max_gicov_m);
	CECL_SET_KERNEL_ARG(dilate_kernel, 1, sizeof(cl_int), (void *) &max_gicov_n);
	CECL_SET_KERNEL_ARG(dilate_kernel, 2, sizeof(cl_int), (void *) &strel_m);
	CECL_SET_KERNEL_ARG(dilate_kernel, 3, sizeof(cl_int), (void *) &strel_n);
	CECL_SET_KERNEL_ARG(dilate_kernel, 4, sizeof(cl_mem), (void *) &c_strel);
	#ifdef USE_IMAGE
	CECL_SET_KERNEL_ARG(dilate_kernel, 5, sizeof(cl_mem), (void *) &device_gicov_image);
	#else
	CECL_SET_KERNEL_ARG(dilate_kernel, 5, sizeof(cl_mem), (void *) &device_gicov);
	#endif
	CECL_SET_KERNEL_ARG(dilate_kernel, 6, sizeof(cl_mem), (void *) &device_img_dilated);

	// Execute the dilation kernel
	error = CECL_ND_RANGE_KERNEL(command_queue, dilate_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
	
	// Check for kernel errors
	error = clFinish(command_queue);
	check_error(error, __FILE__, __LINE__);

	// Copy the result to the host
	// float *host_img_dilated = (cl_float *) CECL_MAP_BUFFER(command_queue, device_img_dilated, CL_TRUE, CL_MAP_READ, 0, max_gicov_mem_size, 0, NULL, NULL, &error);
	float *host_img_dilated = (float*) malloc(max_gicov_mem_size);
	error = CECL_READ_BUFFER(command_queue, device_img_dilated, CL_TRUE, 0, max_gicov_mem_size, host_img_dilated, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);

	// Cleanup memory
	clReleaseMemObject(device_gicov);
	#ifdef USE_IMAGE
	clReleaseMemObject(device_gicov_image);
	#endif
	clReleaseMemObject(device_img_dilated);

	return host_img_dilated;
}


// Chooses the most appropriate GPU on which to execute
void select_device() {
	cl_int error;
  
  cl_platform_id platform_id;

  getOpenCLPlatformIdAndDeviceId(CL_DEVICE_TYPE_GPU, &platform_id, &device);

	// Create an OpenCL context
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id, 0};
	context = CECL_CREATE_CONTEXT_FROM_TYPE(ctxprop, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
	
	// Create a command queue
	command_queue = CECL_CREATE_COMMAND_QUEUE(context, device, 0, &error);
	check_error(error, __FILE__, __LINE__);
		
	return;
}


// Transfers pre-computed constants used by the two kernels to the GPU
void transfer_constants(float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY, int strel_m, int strel_n, float *host_strel) {
	cl_int error;

	// Compute the sizes of the matrices
	unsigned int angle_mem_size = sizeof(float) * NPOINTS;
	unsigned int t_mem_size = sizeof(int) * NCIRCLES * NPOINTS;
	unsigned int strel_mem_size = sizeof(float) * strel_m * strel_n;

	// Allocate device memory and copy the matrices
	c_sin_angle = CECL_BUFFER(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, angle_mem_size, host_sin_angle, &error);
	check_error(error, __FILE__, __LINE__);
	c_cos_angle = CECL_BUFFER(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, angle_mem_size, host_cos_angle, &error);
	check_error(error, __FILE__, __LINE__);
	c_tX = CECL_BUFFER(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, t_mem_size, host_tX, &error);
	check_error(error, __FILE__, __LINE__);
	c_tY = CECL_BUFFER(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, t_mem_size, host_tY, &error);
	check_error(error, __FILE__, __LINE__);
	c_strel = CECL_BUFFER(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, strel_mem_size, host_strel, &error);
	check_error(error, __FILE__, __LINE__);
}
