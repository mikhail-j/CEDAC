/* CUDA EDAC function definitions
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
#include <cuda.h>

#ifndef CUECC_VERSION
#define CUECC_VERSION 1000
#endif /* CUECC_VERSION */

#ifndef CUDA_ECC_H
#define CUDA_ECC_H

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_EDAC_SUCCESS						0
#define CUDA_EDAC_INVALID_ARGUMENT				-1
#define CUDA_EDAC_OUT_OF_MEMORY					-2
#define CUDA_EDAC_PTHREAD_ERROR					-3
#define CUDA_EDAC_MEMORY_OBJECT_ALREADY_IN_USE	-4
#define CUDA_EDAC_MEMORY_OBJECT_NOT_FOUND		-5
#define CUDA_EDAC_INVALID_HSIAO_72_64_VERSION	-6

extern const unsigned char hsiao_22_16_cuda[];
extern const unsigned char hsiao_39_32_cuda[];
extern const unsigned char hsiao_72_64_v1_cuda[];
extern const unsigned char hsiao_72_64_v2_cuda[];

typedef struct cudaECCMemoryObject_st {
	CUdeviceptr data;
	CUdeviceptr parity;

	/*
		errors[0] = total number of single-bit errors detected
		errors[1] = total number of double bit errors detected
	*/
	CUdeviceptr errors;				//in device memory
	uint64_t total_errors[2];		//in host memory
	size_t total_errors_data_size;
	pthread_mutex_t mutex;

	CUdevice device;
	CUcontext context;	//device primary context
	CUmodule module;
	CUfunction generator_kernel;
	CUfunction edac_kernel;
	void* kernel_arguments[4];
	size_t element_size;		//in bytes
	size_t element_count;		//number of elements

	int DEVICE_MULTIPROCESSOR_COUNT;
	int DEVICE_MAX_THREADS_PER_BLOCK;
} cudaECCMemoryObject_t;

typedef struct cudaECCMemoryObjectList_st {
	cudaECCMemoryObject_t* data;
	struct cudaECCMemoryObjectList_st* next;
} cudaECCMemoryObjectList_t;

typedef struct cudaECCHandle_st {
	cudaECCMemoryObjectList_t** MEMORY_ALLOCATIONS;
	pthread_mutex_t MEMORY_ALLOCATIONS_MUTEX;

	char ERRNO_STRING_BUFFER[1024];

	unsigned char PREFERRED_HSIAO_72_64_VERSION;
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
} cudaECCHandle_t;

//locking the EDAC mutex will block EDAC from occurring
int cuECCLockEDACMutex(cudaECCHandle_t* handle);

//unlock EDAC mutex
int cuECCUnlockEDACMutex(cudaECCHandle_t* handle);

int cuECCSetPreferredHsiao_77_22_Version(cudaECCHandle_t* handle, unsigned char version);

int cuECCGetPreferredHsiao_77_22_Version(cudaECCHandle_t* handle, unsigned char* version);

int cuECCSetMemoryScrubbingInterval(cudaECCHandle_t* handle, unsigned long long seconds);

int cuECCGetMemoryScrubbingInterval(cudaECCHandle_t* handle, unsigned long long* seconds);

int cuECCInit(cudaECCHandle_t** handle);

void cuECCDestroy(cudaECCHandle_t* handle);

//if 'memory_object' is NULL, the argument will be ignored
int cuECCAddMemoryObject(cudaECCHandle_t* handle, CUdeviceptr device_memory, cudaECCMemoryObject_t** memory_object);

//cuECCRemoveMemoryObject() will call free() on 'memory_object' on success
int cuECCRemoveMemoryObject(cudaECCHandle_t* handle, cudaECCMemoryObject_t* memory_object);

int cuECCRemoveMemoryObjectWithDevicePointer(cudaECCHandle_t* handle, CUdeviceptr device_memory);

//cuECCUpdateMemoryObject() accepts 'cudaECCMemoryObject_t *' instead of 'CUdeviceptr', avoiding
//the lookup of CUDA memory object.
//Note: Using cuECCUpdateMemoryObject() without locking EDAC mutex creates a race condition.
int cuECCUpdateMemoryObject(cudaECCHandle_t* handle, cudaECCMemoryObject_t* memory_object);

//update CUDA memory object ECC
int cuECCUpdateMemoryObjectWithDevicePointer(cudaECCHandle_t* handle, CUdeviceptr device_memory);

//obtain parity bits returned as a device memory allocation
int cuECCGetMemoryObjectParityBits(cudaECCHandle_t* handle, cudaECCMemoryObject_t* memory_object, CUdeviceptr* parity_memory);

//obtain parity bits returned as a device memory allocation
int cuECCGetMemoryObjectParityBitsWithDevicePointer(cudaECCHandle_t* handle, CUdeviceptr device_memory, CUdeviceptr* parity_memory);

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int cuECCGetTotalErrors(cudaECCHandle_t* handle, cudaECCMemoryObject_t* memory_object, uint64_t* errors, size_t errors_size);

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int cuECCGetTotalErrorsWithDevicePointer(cudaECCHandle_t* handle, CUdeviceptr device_memory, uint64_t* errors, size_t errors_size);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_ECC_H */
