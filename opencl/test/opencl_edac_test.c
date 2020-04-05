/* OpenCL EDAC assertion tests
 *
 * Copyright (C) 2020 Qijia (Michael) Jin
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>

#include <clecc.h>

int main(int argc, char* argv[]) {
	cl_int error_status;
	cl_event event;
	uint64_t total_errors[2];
	unsigned long long delay;
	size_t total_errors_size;

	cl_platform_id platform;
	assert(clGetPlatformIDs(1, &platform, NULL) == CL_SUCCESS);

	size_t driver_version_length;
	assert(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &driver_version_length) == CL_SUCCESS);

	char* driver_version = (char *)malloc(driver_version_length);
	if (driver_version == NULL) {
		perror("malloc()");
		return 1;
	}

	assert(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, driver_version_length, driver_version, NULL) == CL_SUCCESS);

	printf("OpenCL driver version: %s\n", driver_version);

	free(driver_version);

	cl_device_id dev;
	assert(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_SUCCESS);

	uint64_t* A = malloc(sizeof(uint64_t) * 31);
	if (A == NULL) {
		perror("malloc()");
		return 1;
	}
	for (uint64_t i = 0; i < 31; i++) {
		A[i] = i;
	}

	cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &error_status);
	assert(error_status == CL_SUCCESS);

	cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 31 * sizeof(uint64_t), NULL, &error_status);
	assert(error_status == CL_SUCCESS);

	cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 31 * sizeof(uint32_t), NULL, &error_status);
	assert(error_status == CL_SUCCESS);

	cl_mem d_C = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 31 * sizeof(uint16_t), NULL, &error_status);
	assert(error_status == CL_SUCCESS);

#ifdef CL_VERSION_2_0
	cl_command_queue opencl_queue = clCreateCommandQueueWithProperties(ctx, dev, NULL, &error_status);
#else /* OpenCL version < 2.0 */
	cl_command_queue opencl_queue = clCreateCommandQueue(ctx, dev, 0, &error_status);
#endif
	assert(error_status == CL_SUCCESS);

	//copy data from CPU to GPU
    assert(clEnqueueWriteBuffer(opencl_queue, d_A, CL_TRUE, 0, 31 * sizeof(uint64_t), A, 0, NULL, &event) == CL_SUCCESS);
    assert(clWaitForEvents(1, &event) == CL_SUCCESS);

	clECCHandle_t handle;

	clECCMemObject_t mem_A;
	cl_mem parity_A;
	cl_mem parity_B;
	assert(clECCInit(&handle) == 0);

	assert(clECCSetPreferredHsiao_77_22_Version(handle, 2) == 0);

	//default memory scrubbing interval is 300 seconds
	assert(clECCGetMemoryScrubbingInterval(handle, &delay) == 0);
	assert(delay == 300);
	assert(clECCSetMemoryScrubbingInterval(handle, 5) == 0);
	assert(clECCGetMemoryScrubbingInterval(handle, &delay) == 0);
	assert(delay == 5);

	assert(clECCAddMemObject(handle, d_A, opencl_queue, &mem_A) == OPENCL_EDAC_SUCCESS);
	assert(clECCAddMemObject(handle, d_B, opencl_queue, NULL) == OPENCL_EDAC_SUCCESS);
	assert(clECCAddMemObject(handle, d_C, opencl_queue, NULL) == OPENCL_EDAC_SUCCESS);

	//lock EDAC mutex
	assert(clECCLockEDACMutex(handle) == 0);

	assert(clECCUpdateMemObject(handle, mem_A) == 0);

	assert(clECCGetMemObjectParityBits(handle, mem_A, &parity_A) == OPENCL_EDAC_SUCCESS);

	assert(clECCUpdateMemObjectWithCLMem(handle, d_B) == OPENCL_EDAC_SUCCESS);

	assert(clECCGetMemObjectParityBitsWithCLMem(handle, d_B, &parity_B) == OPENCL_EDAC_SUCCESS);

	assert(clECCEDAC(handle, mem_A) == OPENCL_EDAC_SUCCESS);

	//unlock EDAC mutex
	assert(clECCUnlockEDACMutex(handle) == 0);
	
	assert(clECCEDAC(handle, mem_A) == OPENCL_EDAC_SUCCESS);

	assert(clECCGetTotalErrorsSizeWithCLMem(handle, d_A, &total_errors_size) == OPENCL_EDAC_SUCCESS);
	assert(total_errors_size == (2 * sizeof(uint64_t)));

	assert(clECCGetTotalErrorsWithCLMem(handle, d_B, total_errors, 2 * sizeof(uint64_t)) == OPENCL_EDAC_SUCCESS);

	assert(clECCGetTotalErrorsSize(mem_A, &total_errors_size) == OPENCL_EDAC_SUCCESS);
	assert(total_errors_size == (2 * sizeof(uint64_t)));

	assert(clECCGetTotalErrors(handle, mem_A, total_errors, 2 * sizeof(uint64_t)) == OPENCL_EDAC_SUCCESS);

	assert(clECCRemoveMemObjectWithCLMem(handle, d_C) == OPENCL_EDAC_SUCCESS);
	assert(clECCRemoveMemObject(handle, mem_A) == OPENCL_EDAC_SUCCESS);

	clECCDestroy(handle);

	assert(clReleaseCommandQueue(opencl_queue) == CL_SUCCESS);
	assert(clReleaseMemObject(d_C) == CL_SUCCESS);
	assert(clReleaseMemObject(d_B) == CL_SUCCESS);
	assert(clReleaseMemObject(d_A) == CL_SUCCESS);
	assert(clReleaseContext(ctx) == CL_SUCCESS);

	free(A);

	return 0;
}
