/* OpenCL EDAC function definitions
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
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif /* _POSIX_C_SOURCE */

#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <CL/cl.h>

#ifndef CLECC_VERSION
#define CLECC_VERSION 1000
#endif /* CLECC_VERSION */

#ifndef OPENCL_ECC_H
#define OPENCL_ECC_H

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned char hsiao_22_16_opencl[];
extern unsigned char hsiao_39_32_opencl[];
extern unsigned char hsiao_72_64_v1_opencl[];
extern unsigned char hsiao_72_64_v2_opencl[];

#define OPENCL_EDAC_SUCCESS 0
#define OPENCL_EDAC_INVALID_ARGUMENT 1
#define OPENCL_EDAC_OUT_OF_MEMORY 2
#define OPENCL_EDAC_INVALID_MEMORY_PERMISSIONS 3
#define OPENCL_EDAC_PTHREAD_ERROR 4
#define OPENCL_EDAC_MEM_OBJECT_ALREADY_IN_USE 5
#define OPENCL_EDAC_MEM_OBJECT_NOT_FOUND 6
#define OPENCL_EDAC_INVALID_HSIAO_72_64_VERSION 7

typedef struct clECCMemObject_st {
	cl_mem data;
	cl_mem parity;

	/*
		errors[0] = total number of single-bit errors detected
		errors[1] = total number of double bit errors detected
	*/
	cl_mem errors;				//in device memory
	uint64_t total_errors[2];	//in host memory
	size_t total_errors_data_size;
	pthread_mutex_t mutex;
	cl_context context;
	cl_command_queue queue;
	cl_device_id* devices;
	cl_uint device_count;
	size_t element_size;		//in bytes
	uint64_t element_count;		//number of elements
	cl_program program;
	cl_kernel generator_kernel;
	cl_kernel edac_kernel;

	cl_int DEVICE_MAX_COMPUTE_UNITS;
	size_t DEVICE_MAX_WORK_GROUP_SIZE;

	size_t KERNEL_GLOBAL_WORK_SIZE;
} clECCMemObject_t;

typedef struct clECCMemObjectList_st {
	clECCMemObject_t* data;
	struct clECCMemObjectList_st* next;
} clECCMemObjectList_t;

typedef struct clECCHandle_st {
	clECCMemObjectList_t** MEMORY_ALLOCATIONS;
	pthread_mutex_t MEMORY_ALLOCATIONS_MUTEX;

	char ERRNO_STRING_BUFFER[1024];

	uint8_t PREFERRED_HSIAO_72_64_VERSION;
	pthread_mutex_t PREFERRED_HSIAO_72_64_VERSION_MUTEX;

	pthread_mutex_t EDAC_MUTEX;		//mutex for deciding if error detection and correction occurs

	pthread_t EDAC_THREAD;
	pthread_cond_t EDAC_THREAD_WAIT_CONDITION;
	pthread_mutex_t EDAC_THREAD_WAIT_CONDITION_MUTEX;
	pthread_mutex_t EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX;

	unsigned long long EDAC_MEMORY_SCRUBBING_INTERVAL;			//in seconds
	pthread_mutex_t EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX;

	bool IS_ALIVE;
	pthread_mutex_t IS_ALIVE_MUTEX;
} clECCHandle_t;

const char* clGetErrorName(cl_int error);

int clECCLockEDACMutex(clECCHandle_t* handle);

int clECCUnlockEDACMutex(clECCHandle_t* handle);

int clECCSetPreferredHsiao_77_22_Version(clECCHandle_t* handle, uint8_t version);

int clECCGetPreferredHsiao_77_22_Version(clECCHandle_t* handle, uint8_t* version);

int clECCSetMemoryScrubbingInterval(clECCHandle_t* handle, unsigned long long seconds);

int clECCGetMemoryScrubbingInterval(clECCHandle_t* handle, unsigned long long* seconds);

int clECCInit(clECCHandle_t** handle);

void clECCDestroy(clECCHandle_t* handle);

//if 'memory_object' is NULL, the argument will be ignored
int clECCAddMemObject(clECCHandle_t* handle, cl_mem device_memory, cl_command_queue device_queue, clECCMemObject_t** memory_object);

//clECCRemoveMemObject() will call free() on 'memory_object' on success
int clECCRemoveMemObject(clECCHandle_t* handle, clECCMemObject_t* memory_object);

int clECCRemoveMemObjectWithCLMem(clECCHandle_t* handle, cl_mem device_memory);

//clECCUpdateMemObject() accepts 'clECCMemObject_t *' instead of 'cl_mem', avoiding the lookup of OpenCL memory object
//Note: Using clECCUpdateMemObject() without locking EDAC mutex creates a race condition.
int clECCUpdateMemObject(clECCHandle_t* handle, clECCMemObject_t* memory_object);

//update OpenCL memory object ECC
int clECCUpdateMemObjectWithCLMem(clECCHandle_t* handle, cl_mem device_memory);

//obtain parity bits returned as a device memory allocation
int clECCGetMemObjectParityBitsWithCLMem(clECCHandle_t* handle, cl_mem device_memory, cl_mem* parity_memory);

//obtain parity bits returned as a device memory allocation
int clECCGetMemObjectParityBits(clECCHandle_t* handle, clECCMemObject_t* memory_object, cl_mem* parity_memory);

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int clECCGetTotalErrors(clECCHandle_t* handle, clECCMemObject_t* memory_object, uint64_t* errors, size_t errors_size);

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int clECCGetTotalErrorsWithCLMem(clECCHandle_t* handle, cl_mem device_memory, uint64_t* errors, size_t errors_size);

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_ECC_H */
