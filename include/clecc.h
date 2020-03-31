/*
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

#include <stdint.h>
#include <CL/cl.h>

#ifndef CLECC_VERSION
#define CLECC_VERSION 1000
#endif /* CLECC_VERSION */

#ifndef CLECC_H
#define CLECC_H

#ifdef __cplusplus
extern "C" {
#endif

#define OPENCL_EDAC_SUCCESS 0
#define OPENCL_EDAC_INVALID_ARGUMENT 1
#define OPENCL_EDAC_OUT_OF_MEMORY 2
#define OPENCL_EDAC_INVALID_MEMORY_PERMISSIONS 3
#define OPENCL_EDAC_PTHREAD_ERROR 4
#define OPENCL_EDAC_MEM_OBJECT_ALREADY_IN_USE 5
#define OPENCL_EDAC_MEM_OBJECT_NOT_FOUND 6
#define OPENCL_EDAC_INVALID_HSIAO_72_64_VERSION 7

extern unsigned char hsiao_22_16_opencl[];
extern unsigned char hsiao_39_32_opencl[];
extern unsigned char hsiao_72_64_v1_opencl[];
extern unsigned char hsiao_72_64_v2_opencl[];

typedef struct clECCMemObject_st* clECCMemObject_t;

typedef struct clECCHandle_st* clECCHandle_t;

const char* clGetErrorName(cl_int error);

int clECCLockEDACMutex(clECCHandle_t handle);

int clECCUnlockEDACMutex(clECCHandle_t handle);

int clECCSetPreferredHsiao_77_22_Version(clECCHandle_t handle, uint8_t version);

int clECCGetPreferredHsiao_77_22_Version(clECCHandle_t handle, uint8_t* version);

int clECCSetMemoryScrubbingInterval(clECCHandle_t handle, unsigned long long seconds);

int clECCGetMemoryScrubbingInterval(clECCHandle_t handle, unsigned long long* seconds);

int clECCInit(clECCHandle_t* handle);

void clECCDestroy(clECCHandle_t handle);

//if 'memory_object' is NULL, the argument will be ignored
int clECCAddMemObject(clECCHandle_t handle, cl_mem device_memory, cl_command_queue device_queue, clECCMemObject_t* memory_object);

//clECCRemoveMemObject() will call free() on 'memory_object' on success
int clECCRemoveMemObject(clECCHandle_t handle, clECCMemObject_t memory_object);

int clECCRemoveMemObjectWithCLMem(clECCHandle_t handle, cl_mem device_memory);

//clECCUpdateMemObject() accepts 'clECCMemObject_t *' instead of 'cl_mem', avoiding the lookup of OpenCL memory object
//Note: Using clECCUpdateMemObject() without locking EDAC mutex creates a race condition.
int clECCUpdateMemObject(clECCHandle_t handle, clECCMemObject_t memory_object);

//update OpenCL memory object ECC
int clECCUpdateMemObjectWithCLMem(clECCHandle_t handle, cl_mem device_memory);

//obtain parity bits returned as a device memory allocation
int clECCGetMemObjectParityBitsWithCLMem(clECCHandle_t handle, cl_mem device_memory, cl_mem* parity_memory);

//obtain parity bits returned as a device memory allocation
int clECCGetMemObjectParityBits(clECCHandle_t handle, clECCMemObject_t memory_object, cl_mem* parity_memory);

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int clECCGetTotalErrors(clECCHandle_t handle, clECCMemObject_t memory_object, uint64_t* errors, size_t errors_size);

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int clECCGetTotalErrorsWithCLMem(clECCHandle_t handle, cl_mem device_memory, uint64_t* errors, size_t errors_size);

#ifdef __cplusplus
}
#endif

#endif /* CLECC_H */
