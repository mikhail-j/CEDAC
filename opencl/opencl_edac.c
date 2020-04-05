/* OpenCL EDAC functions
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
#include "opencl_edac.h"
#include "opencl_edac_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* clGetErrorName(cl_int error) {
	switch (error) {
		case CL_SUCCESS:
			return "CL_SUCCESS";
		case CL_DEVICE_NOT_FOUND:
			return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:
			return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:
			return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:
			return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:
			return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:
			return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:
			return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:
			return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:
			return "CL_MAP_FAILURE";
#ifdef CL_VERSION_1_1
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
#ifdef CL_VERSION_1_2
		case CL_COMPILE_PROGRAM_FAILURE:
			return "CL_COMPILE_PROGRAM_FAILURE";
		case CL_LINKER_NOT_AVAILABLE:
			return "CL_LINKER_NOT_AVAILABLE";
		case CL_LINK_PROGRAM_FAILURE:
			return "CL_LINK_PROGRAM_FAILURE";
		case CL_DEVICE_PARTITION_FAILED:
			return "CL_DEVICE_PARTITION_FAILED";
		case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
			return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
		case CL_INVALID_VALUE:
			return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:
			return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:
			return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:
			return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:
			return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:
			return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:
			return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:
			return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:
			return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:
			return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:
			return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:
			return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:
			return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:
			return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:
			return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:
			return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:
			return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:
			return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:
			return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:
			return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:
			return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:
			return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:
			return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:
			return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:
			return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:
			return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:
			return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:
			return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:
			return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:
			return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:
			return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifdef CL_VERSION_1_1
		case CL_INVALID_PROPERTY:
			return "CL_INVALID_PROPERTY";
#endif
#ifdef CL_VERSION_1_2
		case CL_INVALID_IMAGE_DESCRIPTOR:
			return "CL_INVALID_IMAGE_DESCRIPTOR";
		case CL_INVALID_COMPILER_OPTIONS:
			return "CL_INVALID_COMPILER_OPTIONS";
		case CL_INVALID_LINKER_OPTIONS:
			return "CL_INVALID_LINKER_OPTIONS";
		case CL_INVALID_DEVICE_PARTITION_COUNT:
			return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
#ifdef CL_VERSION_2_0
		case CL_INVALID_PIPE_SIZE:
			return "CL_INVALID_PIPE_SIZE";
		case CL_INVALID_DEVICE_QUEUE:
			return "CL_INVALID_DEVICE_QUEUE";
#endif
#ifdef CL_VERSION_2_2
		case CL_INVALID_SPEC_ID:
			return "CL_INVALID_SPEC_ID";
		case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
			return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif
		default: 
//			printf("error: unknown error message, %d!\n", error);
		;
	}
	return NULL;
}

static void free_opencl_ecc_memory_object_list(char* string_buffer, clECCMemObjectList_t** objects) {
	if (objects == NULL) {
		return;
	}
	int status;

	clECCMemObjectList_t* current = *objects;
	clECCMemObjectList_t* next;

	while (current != NULL) {
		assert(clReleaseMemObject(current->data->errors) == CL_SUCCESS);
		assert(clReleaseMemObject(current->data->parity) == CL_SUCCESS);
		assert(clReleaseKernel(current->data->edac_kernel) == CL_SUCCESS);
		assert(clReleaseKernel(current->data->generator_kernel) == CL_SUCCESS);
		assert(clReleaseProgram(current->data->program) == CL_SUCCESS);

		//destroy corresponding memory object mutex
		status = pthread_mutex_destroy(&(current->data->mutex));
		if (status != 0) {
			while (status == EBUSY) {
				status = pthread_mutex_destroy(&(current->data->mutex));
			}
			if (status != 0 && status != EBUSY) {
				assert(strerror_r(status, string_buffer, 1024) == 0);
				printf("free_opencl_ecc_memory_object_list(): pthread_mutex_destroy(): error: %s\n", string_buffer);
				
				//encountered unrecoverable error, exit immediately
				exit(1);
			}
		}

		//free memory allocations
		free(current->data->devices);
		free(current->data);

		//set next node
		next = current->next;
		free(current);
		current = next;
	}

	//prevent access to freed memory by setting reference of (*object) to NULL
	*objects = NULL;

	return;
}

int clECCInit(clECCHandle_t** handle) {
	if (handle == NULL) {
		return 1;
	}

	*handle = (clECCHandle_t *)malloc(sizeof(clECCHandle_t));
	if ((*handle) == NULL) {
		perror("malloc()");
		errno = 0;
		return 2;
	}

	(*handle)->MEMORY_ALLOCATIONS = (clECCMemObjectList_t **)malloc(sizeof(clECCMemObjectList_t *));
	if ((*handle)->MEMORY_ALLOCATIONS == NULL) {
		perror("malloc()");
		errno = 0;

		//free memory allocations
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return 2;
	}
	//initialize empty list
	*((*handle)->MEMORY_ALLOCATIONS) = NULL;

	int status = pthread_mutex_init(&((*handle)->MEMORY_ALLOCATIONS_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return 4;
	}

	status = pthread_mutex_init(&((*handle)->IS_ALIVE_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return 4;
	}

	status = pthread_cond_init(&((*handle)->EDAC_THREAD_WAIT_CONDITION), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCInit(): pthread_cond_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->IS_ALIVE_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return 4;
	}

	status = pthread_mutex_init(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized conditions
		assert(pthread_cond_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION)) == 0);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->IS_ALIVE_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return 4;
	}

	status = pthread_mutex_init(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized conditions
		assert(pthread_cond_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION)) == 0);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->IS_ALIVE_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return 4;
	}

	status = pthread_mutex_init(&((*handle)->PREFERRED_HSIAO_72_64_VERSION_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized conditions
		assert(pthread_cond_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION)) == 0);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->IS_ALIVE_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return 4;
	}

	status = pthread_mutex_init(&((*handle)->EDAC_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized conditions
		assert(pthread_cond_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION)) == 0);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->PREFERRED_HSIAO_72_64_VERSION_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->IS_ALIVE_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return 4;
	}

	status = pthread_mutex_init(&((*handle)->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized conditions
		assert(pthread_cond_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION)) == 0);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->EDAC_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->PREFERRED_HSIAO_72_64_VERSION_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->IS_ALIVE_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return -3;
	}

	//initialize new thread
	status = pthread_create(&((*handle)->EDAC_THREAD), NULL, handle_opencl_edac_thread, (void *)(*handle));
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized conditions
		assert(pthread_cond_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION)) == 0);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->EDAC_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->PREFERRED_HSIAO_72_64_VERSION_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->IS_ALIVE_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return 4;
	}

	(*handle)->IS_ALIVE = true;

	//use Hsiao(72, 64) version 1 codes by default for 64-bit EDAC
	(*handle)->PREFERRED_HSIAO_72_64_VERSION = 1;

	//set memory scrubbing interval to 5 minutes
	(*handle)->EDAC_MEMORY_SCRUBBING_INTERVAL = 300;

	return 0;
}

//calling clECCDestroy() on previous destroyed handle will produce unexpected behavior
void clECCDestroy(clECCHandle_t* handle) {
	if (handle == NULL) {
		return;
	}

	int status;

	status = pthread_mutex_lock(&(handle->IS_ALIVE_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_mutex_lock(): %s\n", handle->ERRNO_STRING_BUFFER);
		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	handle->IS_ALIVE = false;

	status = pthread_mutex_unlock(&(handle->IS_ALIVE_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_mutex_unlock(): %s\n", handle->ERRNO_STRING_BUFFER);
		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//check if wait condition mutex mutex is locked every 0.01 seconds
	status = 0;
	while (status == 0) {
		status = pthread_mutex_trylock(&(handle->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX));
		if (status == 0) {
			//unlock mutex for wait condition
			status = pthread_mutex_unlock(&(handle->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCDestroy(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
				//unable to unlock mutex lock for wait condition, force exit
				exit(1);
			}

			struct timespec t;
			double t0, t1;

			status = clock_gettime(CLOCK_REALTIME, &t);
			if (status != 0) {
				perror("clock_gettime()");
				errno = 0;

				//unable to obtain time for delay, force exit
				exit(1);
			}
			t0 = (((double)t.tv_sec) + (t.tv_nsec * 0.000000001));
			t1 = t0;

			//delay the next pthread_mutex_trylock() by 0.01 seconds
			while (t1 - t0 < 0.01) {
				status = clock_gettime(CLOCK_REALTIME, &t);
				if (status != 0) {
					perror("clock_gettime()");
					errno = 0;
	
					//unable to obtain time for delay, force exit
					exit(1);
				}
				t1 = (((double)t.tv_sec) + (t.tv_nsec * 0.000000001));
			}
		}
		else if (status == EBUSY) {
			//wait condition mutex is already locked, break the current while loop
			break;
		}
		else {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("clECCDestroy(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
			//unable to obtain mutex lock for wait condition, force exit
			exit(1);
		}
	}

	//wake up EDAC thread immediately
	status = pthread_cond_signal(&(handle->EDAC_THREAD_WAIT_CONDITION));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_cond_signal(): %s\n", handle->ERRNO_STRING_BUFFER);
		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//join thread first
	status = pthread_join(handle->EDAC_THREAD, NULL);
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//clECCDestroy() maybe called without properly unlocking EDAC mutex beforehand
	status = pthread_mutex_destroy(&(handle->EDAC_MUTEX));
	if (status == 0) {}
	else if (status == EBUSY) {
		status = pthread_mutex_unlock(&(handle->EDAC_MUTEX));
		if (status != 0) {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("clECCDestroy(): pthread_mutex_destroy(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);	
			//encountered unrecoverable error, exit immediately
			exit(1);
		}

		status = pthread_mutex_destroy(&(handle->EDAC_MUTEX));
		if (status != 0) {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("clECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
			//encountered unrecoverable error, exit immediately
			exit(1);
		}
		//successfully destroyed EDAC mutex
	}
	else {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->PREFERRED_HSIAO_72_64_VERSION_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->EDAC_THREAD_WAIT_CONDITION_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_cond_destroy(&(handle->EDAC_THREAD_WAIT_CONDITION));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_cond_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->IS_ALIVE_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}


	//free all memory allocations in the linked list
	free_opencl_ecc_memory_object_list(handle->ERRNO_STRING_BUFFER, handle->MEMORY_ALLOCATIONS);

	//free memory allocations
	free(handle->MEMORY_ALLOCATIONS);
	free(handle);

	return;
}

//if 'memory_object' is NULL, the argument will be ignored
int clECCAddMemObject(clECCHandle_t* handle, cl_mem device_memory, cl_command_queue device_queue, clECCMemObject_t** memory_object) {
	if (handle == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	clECCMemObjectList_t* node;

	//check if memory object is already in handle
	node = *(handle->MEMORY_ALLOCATIONS);

	while (node != NULL) {
		if (node->data->data == device_memory) {
			//memory object was previously added with clECCAddMemObject()
			return 5;
		}

		//set next node
		node = node->next;
	}

	clECCMemObject_t* new_memory_object = (clECCMemObject_t *)malloc(sizeof(clECCMemObject_t));
	if (new_memory_object == NULL) {
		perror("malloc()");
		errno = 0;
		return 2;
	}
	new_memory_object->devices = NULL;

	node = (clECCMemObjectList_t *)malloc(sizeof(clECCMemObjectList_t));
	if (node == NULL) {
		perror("malloc()");
		errno = 0;

		//free memory allocations
		free(new_memory_object);

		return 2;
	}
	node->data = new_memory_object;
	node->next = *(handle->MEMORY_ALLOCATIONS);

	//assign arguments in new_memory_object
	new_memory_object->data = device_memory;
	new_memory_object->queue = device_queue;

	new_memory_object->total_errors_data_size = 2 * sizeof(uint64_t);

	cl_mem_flags flags;
	cl_int status;
	size_t device_memory_size;
	cl_context queue_context;

	status = clGetMemObjectInfo(device_memory, CL_MEM_CONTEXT, sizeof(cl_context), &(new_memory_object->context), NULL);
	if (status != CL_SUCCESS) {
		//free memory allocations
		free(node);
		free(new_memory_object);
		return status;
	}

	status =  clGetCommandQueueInfo(device_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &queue_context, NULL);
	if (status != CL_SUCCESS) {
		//free memory allocations
		free(node);
		free(new_memory_object);
		return status;
	}

	//verify context association between memory object
	if (new_memory_object->context != queue_context) {
		//free memory allocations
		free(node);
		free(new_memory_object);

		printf("clECCAddMemObject(): error: the command queue context does not match given device memory context!\n");
		return 1;
	}

	status = clGetMemObjectInfo(device_memory, CL_MEM_FLAGS, sizeof(cl_mem_flags), &flags, NULL);
	if (status != CL_SUCCESS) {
		//free memory allocations
		free(node);
		free(new_memory_object);
		return status;
	}

	//check if device memory has read and write permissions
	if ((flags & CL_MEM_READ_WRITE) != CL_MEM_READ_WRITE) {
		//free memory allocations
		free(node);
		free(new_memory_object);

		printf("clECCAddMemObject(): error: device memory allocation must be created with 'CL_MEM_READ_WRITE'!\n");
		return 3;
	}

	status = clGetMemObjectInfo(device_memory, CL_MEM_SIZE, sizeof(size_t), &device_memory_size, NULL);
	if (status != CL_SUCCESS) {
		//free memory allocations
		free(node);
		free(new_memory_object);
		return status;
	}

	status = clGetContextInfo(new_memory_object->context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &(new_memory_object->device_count), NULL);
	if (status != CL_SUCCESS) {
		//free memory allocations
		free(node);
		free(new_memory_object);
		return status;
	}

	new_memory_object->devices = (cl_device_id *)malloc(new_memory_object->device_count * sizeof(cl_device_id));
	if (new_memory_object->devices == NULL) {
		//free memory allocations
		free(node);
		free(new_memory_object);
		return 2;
	}

	status = clGetContextInfo(new_memory_object->context,
							CL_CONTEXT_DEVICES,
							new_memory_object->device_count * sizeof(cl_device_id),
							new_memory_object->devices,
							NULL);
	if (status != CL_SUCCESS) {
		//free memory allocations
		free(node);
		free(new_memory_object->devices);
		free(new_memory_object);
		return status;
	}

	status = clGetDeviceInfo(*(new_memory_object->devices),
							CL_DEVICE_MAX_COMPUTE_UNITS,
							sizeof(cl_uint),
							&(new_memory_object->DEVICE_MAX_COMPUTE_UNITS),
							NULL);
	if (status != CL_SUCCESS) {
		//free memory allocations
		free(node);
		free(new_memory_object->devices);
		free(new_memory_object);
		return status;
	}

	status = clGetDeviceInfo(*(new_memory_object->devices),
							CL_DEVICE_MAX_WORK_GROUP_SIZE,
							sizeof(size_t),
							&(new_memory_object->DEVICE_MAX_WORK_GROUP_SIZE),
							NULL);
	if (status != CL_SUCCESS) {
		//free memory allocations
		free(node);
		free(new_memory_object->devices);
		free(new_memory_object);
		return status;
	}
	new_memory_object->KERNEL_GLOBAL_WORK_SIZE = new_memory_object->DEVICE_MAX_COMPUTE_UNITS * new_memory_object->DEVICE_MAX_WORK_GROUP_SIZE;
	

	//ECC only implemented for 16-bit, 32-bit, and 64-bit data types
	if ((device_memory_size % sizeof(uint64_t)) == 0) {
		new_memory_object->element_count = device_memory_size / sizeof(uint64_t);
		new_memory_object->element_size = sizeof(uint64_t);
		uint8_t hsiao_72_64_version;

		//throw assertion error if mutex is not usable
		assert(clECCGetPreferredHsiao_77_22_Version(handle, &hsiao_72_64_version) == 0);

		if (hsiao_72_64_version == 1) {
			char * sources[] = { hsiao_72_64_v1_opencl };
			new_memory_object->program = clCreateProgramWithSource(new_memory_object->context, 1, (const char **)sources, NULL, &status);
			if (status != CL_SUCCESS) {
				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);
				return status;
			}
			status = clBuildProgram(new_memory_object->program, 1, new_memory_object->devices, NULL, NULL, NULL);
			if (status == CL_BUILD_PROGRAM_FAILURE) {
				size_t error_log_size;

				assert(clGetProgramBuildInfo(new_memory_object->program, *(new_memory_object->devices), CL_PROGRAM_BUILD_LOG, 0, NULL, &error_log_size) == CL_SUCCESS);
				
				char * error_log = (char *)malloc(error_log_size);
				if (error_log == NULL) {
					perror("clECCAddMemObject(): clBuildProgram(): malloc()");
					errno = 0;

					assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

					//free memory allocations
					free(node);
					free(new_memory_object->devices);
					free(new_memory_object);

					return 1;
				}

				assert(clGetProgramBuildInfo(new_memory_object->program, *(new_memory_object->devices), CL_PROGRAM_BUILD_LOG, error_log_size, error_log, NULL) == CL_SUCCESS);
				printf("clECCAddMemObject(): clGetProgramBuildInfo(): %s\n", error_log);

				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);

				free(error_log);
				return status;
			}
			else if (status != CL_SUCCESS) {
				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);
				return status;
			}
			new_memory_object->generator_kernel = clCreateKernel(new_memory_object->program, "generate_parity_hsiao_72_64_v1", &status);
			if (status != CL_SUCCESS) {
				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);
				return status;
			}
			new_memory_object->edac_kernel = clCreateKernel(new_memory_object->program, "edac_hsiao_72_64_v1", &status);
			if (status != CL_SUCCESS) {
				assert(clReleaseKernel(new_memory_object->generator_kernel) == CL_SUCCESS);
				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);
				return status;
			}
		}
		else if (hsiao_72_64_version == 2) {
			char * sources[] = { hsiao_72_64_v2_opencl };
			new_memory_object->program = clCreateProgramWithSource(new_memory_object->context, 1, (const char **)sources, NULL, &status);
			if (status != CL_SUCCESS) {
				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);
				return status;
			}
			status = clBuildProgram(new_memory_object->program, 1, new_memory_object->devices, NULL, NULL, NULL);
			if (status == CL_BUILD_PROGRAM_FAILURE) {
				size_t error_log_size;

				assert(clGetProgramBuildInfo(new_memory_object->program, *(new_memory_object->devices), CL_PROGRAM_BUILD_LOG, 0, NULL, &error_log_size) == CL_SUCCESS);
				
				char * error_log = (char *)malloc(error_log_size);
				if (error_log == NULL) {
					perror("clECCAddMemObject(): clBuildProgram(): malloc()");
					errno = 0;

					assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

					//free memory allocations
					free(node);
					free(new_memory_object->devices);
					free(new_memory_object);

					return 1;
				}

				assert(clGetProgramBuildInfo(new_memory_object->program, *(new_memory_object->devices), CL_PROGRAM_BUILD_LOG, error_log_size, error_log, NULL) == CL_SUCCESS);
				printf("clECCAddMemObject(): clGetProgramBuildInfo(): %s\n", error_log);

				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);

				free(error_log);
				return status;
			}
			else if (status != CL_SUCCESS) {
				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);
				return status;
			}
			new_memory_object->generator_kernel = clCreateKernel(new_memory_object->program, "generate_parity_hsiao_72_64_v2", &status);
			if (status != CL_SUCCESS) {
				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);
				return status;
			}
			new_memory_object->edac_kernel = clCreateKernel(new_memory_object->program, "edac_hsiao_72_64_v2", &status);
			if (status != CL_SUCCESS) {
				assert(clReleaseKernel(new_memory_object->generator_kernel) == CL_SUCCESS);
				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);
				return status;
			}
		}
		else {
			printf("clECCAddMemObject(): error: found invalid 'PREFERRED_HSIAO_72_64_VERSION' value, %d!\n", hsiao_72_64_version);

			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);

			return 7;
		}
	}
	else if ((device_memory_size % sizeof(uint32_t)) == 0) {
		new_memory_object->element_count = device_memory_size / sizeof(uint32_t);
		new_memory_object->element_size = sizeof(uint32_t);

		char * sources[] = { hsiao_39_32_opencl };
		new_memory_object->program = clCreateProgramWithSource(new_memory_object->context, 1, (const char **)sources, NULL, &status);
		if (status != CL_SUCCESS) {
			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);
			return status;
		}
		status = clBuildProgram(new_memory_object->program, 1, new_memory_object->devices, NULL, NULL, NULL);
		if (status == CL_BUILD_PROGRAM_FAILURE) {
			size_t error_log_size;

			assert(clGetProgramBuildInfo(new_memory_object->program, *(new_memory_object->devices), CL_PROGRAM_BUILD_LOG, 0, NULL, &error_log_size) == CL_SUCCESS);
			
			char * error_log = (char *)malloc(error_log_size);
			if (error_log == NULL) {
				perror("clECCAddMemObject(): clBuildProgram(): malloc()");
				errno = 0;

				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);

				return 1;
			}

			assert(clGetProgramBuildInfo(new_memory_object->program, *(new_memory_object->devices), CL_PROGRAM_BUILD_LOG, error_log_size, error_log, NULL) == CL_SUCCESS);
			printf("clECCAddMemObject(): clGetProgramBuildInfo(): %s\n", error_log);

			assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);

			free(error_log);
			return status;
		}
		else if (status != CL_SUCCESS) {
			assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);
			return status;
		}
		new_memory_object->generator_kernel = clCreateKernel(new_memory_object->program, "generate_parity_hsiao_39_32", &status);
		if (status != CL_SUCCESS) {
			assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);
			return status;
		}
		new_memory_object->edac_kernel = clCreateKernel(new_memory_object->program, "edac_hsiao_39_32", &status);
		if (status != CL_SUCCESS) {
			assert(clReleaseKernel(new_memory_object->generator_kernel) == CL_SUCCESS);
			assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);
			return status;
		}
	}
	else if ((device_memory_size % sizeof(uint16_t)) == 0) {
		new_memory_object->element_count = device_memory_size / sizeof(uint16_t);
		new_memory_object->element_size = sizeof(uint16_t);

		char * sources[] = { hsiao_22_16_opencl };
		new_memory_object->program = clCreateProgramWithSource(new_memory_object->context, 1, (const char **)sources, NULL, &status);
		if (status != CL_SUCCESS) {
			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);
			return status;
		}
		status = clBuildProgram(new_memory_object->program, 1, new_memory_object->devices, NULL, NULL, NULL);
		if (status == CL_BUILD_PROGRAM_FAILURE) {
			size_t error_log_size;

			assert(clGetProgramBuildInfo(new_memory_object->program, *(new_memory_object->devices), CL_PROGRAM_BUILD_LOG, 0, NULL, &error_log_size) == CL_SUCCESS);
			
			char * error_log = (char *)malloc(error_log_size);
			if (error_log == NULL) {
				perror("clECCAddMemObject(): clBuildProgram(): malloc()");
				errno = 0;

				assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object->devices);
				free(new_memory_object);

				return 1;
			}

			assert(clGetProgramBuildInfo(new_memory_object->program, *(new_memory_object->devices), CL_PROGRAM_BUILD_LOG, error_log_size, error_log, NULL) == CL_SUCCESS);
			printf("clECCAddMemObject(): clGetProgramBuildInfo(): %s\n", error_log);

			assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);

			free(error_log);
			return status;
		}
		else if (status != CL_SUCCESS) {
			assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);
			return status;
		}
		new_memory_object->generator_kernel = clCreateKernel(new_memory_object->program, "generate_parity_hsiao_22_16", &status);
		if (status != CL_SUCCESS) {
			assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);
			return status;
		}
		new_memory_object->edac_kernel = clCreateKernel(new_memory_object->program, "edac_hsiao_22_16", &status);
		if (status != CL_SUCCESS) {
			assert(clReleaseKernel(new_memory_object->generator_kernel) == CL_SUCCESS);
			assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object->devices);
			free(new_memory_object);
			return status;
		}
	}
	else {
		//free memory allocations
		free(node);
		free(new_memory_object->devices);
		free(new_memory_object);

		printf("clECCAddMemObject(): error: encountered unsupported device memory size, %zu!\n", device_memory_size);
		return status;
	}

	//allocate parity bits in device memory
	new_memory_object->parity = clCreateBuffer(new_memory_object->context,
						CL_MEM_READ_WRITE,
						(device_memory_size / (new_memory_object->element_size)) * sizeof(uint8_t),
						NULL,
						&status);
	if (status != CL_SUCCESS) {
		printf("clECCAddMemObject(): clCreateBuffer(): error: %s\n", clGetErrorName(status));

		assert(clReleaseKernel(new_memory_object->edac_kernel) == CL_SUCCESS);
		assert(clReleaseKernel(new_memory_object->generator_kernel) == CL_SUCCESS);
		assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

		//free memory allocations
		free(node);
		free(new_memory_object->devices);
		free(new_memory_object);
		return status;
	}

	//allocate error counter in device memory
	new_memory_object->errors = clCreateBuffer(new_memory_object->context,
						CL_MEM_READ_WRITE,
						2 * sizeof(uint64_t),
						NULL,
						&status);
	if (status != CL_SUCCESS) {
		printf("clECCAddMemObject(): clCreateBuffer(): error: %s\n", clGetErrorName(status));

		assert(clReleaseMemObject(new_memory_object->parity) == CL_SUCCESS);
		assert(clReleaseKernel(new_memory_object->edac_kernel) == CL_SUCCESS);
		assert(clReleaseKernel(new_memory_object->generator_kernel) == CL_SUCCESS);
		assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

		//free memory allocations
		free(node);
		free(new_memory_object->devices);
		free(new_memory_object);
		return status;
	}

	//initialize error count variables (host memory) to 0
	new_memory_object->total_errors[0] = 0;
	new_memory_object->total_errors[1] = 0;

	cl_event event;

    //copy data from host to device
    assert(clEnqueueWriteBuffer(new_memory_object->queue, new_memory_object->errors, CL_TRUE, 0, 2 * sizeof(uint64_t), new_memory_object->total_errors, 0, NULL, &event) == CL_SUCCESS);
    assert(clWaitForEvents(1, &event) == CL_SUCCESS);

	//assign OpenCL kernel arguments
	assert(clSetKernelArg(new_memory_object->generator_kernel, 0, sizeof(cl_mem), (void*)&(new_memory_object->data)) == CL_SUCCESS);
	assert(clSetKernelArg(new_memory_object->generator_kernel, 1, sizeof(cl_mem), (void*)&(new_memory_object->parity)) == CL_SUCCESS);
	assert(clSetKernelArg(new_memory_object->generator_kernel, 2, sizeof(uint64_t), (void*)&(new_memory_object->element_count)) == CL_SUCCESS);

	assert(clSetKernelArg(new_memory_object->edac_kernel, 0, sizeof(cl_mem), (void*)&(new_memory_object->data)) == CL_SUCCESS);
	assert(clSetKernelArg(new_memory_object->edac_kernel, 1, sizeof(cl_mem), (void*)&(new_memory_object->parity)) == CL_SUCCESS);
	assert(clSetKernelArg(new_memory_object->edac_kernel, 2, sizeof(uint64_t), (void*)&(new_memory_object->element_count)) == CL_SUCCESS);
	assert(clSetKernelArg(new_memory_object->edac_kernel, 3, sizeof(cl_mem), (void*)&(new_memory_object->errors)) == CL_SUCCESS);

	//use generator kernel to generate parity bits
	assert(clEnqueueNDRangeKernel(new_memory_object->queue,
                                new_memory_object->generator_kernel,
                                1,
                                NULL,
                                &(new_memory_object->KERNEL_GLOBAL_WORK_SIZE),
                                &(new_memory_object->DEVICE_MAX_WORK_GROUP_SIZE),
                                0,
                                NULL,
                                &event) == CL_SUCCESS);
    assert(clWaitForEvents(1, &event) == CL_SUCCESS);


	int mutex_status = pthread_mutex_init(&(new_memory_object->mutex), NULL);
	if (mutex_status != 0) {
		assert(strerror_r(mutex_status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCAddMemObject(): pthread_mutex_init(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		assert(clReleaseMemObject(new_memory_object->errors) == CL_SUCCESS);
		assert(clReleaseMemObject(new_memory_object->parity) == CL_SUCCESS);
		assert(clReleaseKernel(new_memory_object->edac_kernel) == CL_SUCCESS);
		assert(clReleaseKernel(new_memory_object->generator_kernel) == CL_SUCCESS);
		assert(clReleaseProgram(new_memory_object->program) == CL_SUCCESS);

		//free memory allocations
		free(node);
		free(new_memory_object->devices);
		free(new_memory_object);

		return 4;
	}

	//push 'node' into linked list
	*(handle->MEMORY_ALLOCATIONS) = node;

	if (memory_object != NULL) {
		*memory_object = new_memory_object;
	}

	return 0;
}

//clECCRemoveMemObject() will call free() on 'memory_object' on success
int clECCRemoveMemObject(clECCHandle_t* handle, clECCMemObject_t* memory_object) {
	if (handle == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for OpenCL memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCRemoveMemObject(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	clECCMemObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	clECCMemObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data == memory_object) {
			assert(clReleaseMemObject(current->data->errors) == CL_SUCCESS);
			assert(clReleaseMemObject(current->data->parity) == CL_SUCCESS);
			assert(clReleaseKernel(current->data->edac_kernel) == CL_SUCCESS);
			assert(clReleaseKernel(current->data->generator_kernel) == CL_SUCCESS);
			assert(clReleaseProgram(current->data->program) == CL_SUCCESS);

			//destroy corresponding memory object mutex
			status = pthread_mutex_destroy(&(current->data->mutex));
			if (status != 0) {
				while (status == EBUSY) {
					//retry pthread_mutex_destroy() on the mutex if another process was using the mutex
					status = pthread_mutex_destroy(&(current->data->mutex));
				}
				if (status != 0 && status != EBUSY) {
					assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
					printf("clECCRemoveMemObject(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
					
					//encountered unrecoverable error, exit immediately
					exit(1);
				}
			}

			//update linked list by removing memory object node
			if (previous == NULL) {
				*(handle->MEMORY_ALLOCATIONS) = current->next;
			}
			else {
				previous->next = current->next;
			}

			//free memory allocations
			free(current->data->devices);
			free(current->data);
			free(current);

			//unlock mutex for OpenCL memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCRemoveMemObject(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				return 4;
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for OpenCL memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCRemoveMemObject(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	//the memory object could not be found in the given handle
	return 6;
}

int clECCRemoveMemObjectWithCLMem(clECCHandle_t* handle, cl_mem device_memory) {
	if (handle == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for OpenCL memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCRemoveMemObjectWithCLMem(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	clECCMemObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	clECCMemObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			assert(clReleaseMemObject(current->data->errors) == CL_SUCCESS);
			assert(clReleaseMemObject(current->data->parity) == CL_SUCCESS);
			assert(clReleaseKernel(current->data->edac_kernel) == CL_SUCCESS);
			assert(clReleaseKernel(current->data->generator_kernel) == CL_SUCCESS);
			assert(clReleaseProgram(current->data->program) == CL_SUCCESS);

			//destroy corresponding memory object mutex
			status = pthread_mutex_destroy(&(current->data->mutex));
			if (status != 0) {
				while (status == EBUSY) {
					//retry pthread_mutex_destroy() on the mutex if another process was using the mutex
					status = pthread_mutex_destroy(&(current->data->mutex));
				}
				if (status != 0 && status != EBUSY) {
					assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
					printf("clECCRemoveMemObjectWithCLMem(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
					
					//encountered unrecoverable error, exit immediately
					exit(1);
				}
			}

			//update linked list by removing memory object node
			if (previous == NULL) {
				*(handle->MEMORY_ALLOCATIONS) = current->next;
			}
			else {
				previous->next = current->next;
			}

			//free memory allocations
			free(current->data->devices);
			free(current->data);
			free(current);

			//unlock mutex for OpenCL memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCRemoveMemObjectWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				return 4;
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for OpenCL memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCRemoveMemObjectWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	//the memory object could not be found in the given handle
	return 6;
}

//clECCUpdateMemObject() accepts 'clECCMemObject_t *' instead of 'cl_mem', avoiding the lookup of OpenCL memory object
//Using clECCUpdateMemObject() without locking EDAC mutex creates a race condition.
int clECCUpdateMemObject(clECCHandle_t* handle, clECCMemObject_t* memory_object) {
	if (handle == NULL || memory_object == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;
	cl_event event;

	//use generator kernel to generate parity bits
	status = clEnqueueNDRangeKernel(memory_object->queue,
                                memory_object->generator_kernel,
                                1,
                                NULL,
                                &(memory_object->KERNEL_GLOBAL_WORK_SIZE),
                                &(memory_object->DEVICE_MAX_WORK_GROUP_SIZE),
                                0,
                                NULL,
                                &event);
    if (status != CL_SUCCESS) {
    	return status;
    }

    status = clWaitForEvents(1, &event);
    //immediately return OpenCL error
	return status;
}

//update OpenCL memory object ECC
int clECCUpdateMemObjectWithCLMem(clECCHandle_t* handle, cl_mem device_memory) {
	if (handle == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;
	cl_event event;

	//lock mutex for OpenCL memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCUpdateMemObjectWithCLMem(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	clECCMemObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	clECCMemObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			//use generator kernel to generate parity bits
			status = clEnqueueNDRangeKernel(current->data->queue,
    		                            current->data->generator_kernel,
    		                            1,
    		                            NULL,
    		                            &(current->data->KERNEL_GLOBAL_WORK_SIZE),
    		                            &(current->data->DEVICE_MAX_WORK_GROUP_SIZE),
    		                            0,
    		                            NULL,
    		                            &event);
    		if (status != CL_SUCCESS) {
    			//unlock mutex for OpenCL memory allocations
				int mutex_status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
				if (mutex_status != 0) {
					assert(strerror_r(mutex_status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
					printf("clECCUpdateMemObjectWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
					
					//encountered unrecoverable error, exit immediately
					exit(1);
				}

				//return OpenCL error
				return status;
    		}
    		
    		//wait for OpenCL kernel to finish
    		status = clWaitForEvents(1, &event);
    		if (status != CL_SUCCESS) {
    			//unlock mutex for OpenCL memory allocations
				int mutex_status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
				if (mutex_status != 0) {
					assert(strerror_r(mutex_status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
					printf("clECCUpdateMemObjectWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
					
					//encountered unrecoverable error, exit immediately
					exit(1);
				}

				//return OpenCL error
				return status;
    		}

			//unlock mutex for OpenCL memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCUpdateMemObjectWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//encountered unrecoverable error, exit immediately
				exit(1);
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for OpenCL memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCUpdateMemObjectWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//the memory object could not be found in the given handle
	return 6;
}

//obtain parity bits returned as a device memory allocation
int clECCGetMemObjectParityBits(clECCHandle_t* handle, clECCMemObject_t* memory_object, cl_mem* parity_memory) {
	if (handle == NULL
		|| memory_object == NULL
		|| parity_memory == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock memory object mutex
	status = pthread_mutex_lock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetMemObjectParityBits(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to obtain mutex lock for memory object, force exit
		exit(1);
	}

	//assign associated parity bits OpenCL memory to dereferenced 'parity_memory' pointer
	*parity_memory = memory_object->parity;

	//unlock memory object mutex
	status = pthread_mutex_unlock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetMemObjectParityBits(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to unlock mutex for memory object, force exit
		exit(1);
	}

	//the memory object could not be found in the given handle
	return 0;
}

//obtain parity bits returned as a device memory allocation
int clECCGetMemObjectParityBitsWithCLMem(clECCHandle_t* handle, cl_mem device_memory, cl_mem* parity_memory) {
	if (handle == NULL || parity_memory == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for OpenCL memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetMemObjectParityBitsWithCLMem(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	clECCMemObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	clECCMemObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			//lock corresponding memory object mutex
			status = pthread_mutex_lock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCGetMemObjectParityBitsWithCLMem(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to obtain mutex lock for memory object, force exit
				exit(1);
			}

			//assign associated parity bits OpenCL memory to dereferenced 'parity_memory' pointer
			*parity_memory = current->data->parity;
			
			//unlock corresponding memory object mutex
			status = pthread_mutex_unlock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCGetMemObjectParityBitsWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to unlock mutex for memory object, force exit
				exit(1);
			}

			//unlock mutex for OpenCL memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCGetMemObjectParityBitsWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				return 4;
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for OpenCL memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetMemObjectParityBitsWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	//the memory object could not be found in the given handle
	return 6;
}

int clECCSetPreferredHsiao_77_22_Version(clECCHandle_t* handle, uint8_t version) {
	if (handle == NULL || (version != 1 && version != 2)) {
		return 1;
	}
	//check if handle is valid by dereferencing the handle pointer
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for 'PREFERRED_HSIAO_72_64_VERSION'
	status = pthread_mutex_lock(&(handle->PREFERRED_HSIAO_72_64_VERSION_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCSetPreferredHsiao_77_22_Version(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	handle->PREFERRED_HSIAO_72_64_VERSION = version;

	//unlock mutex for 'PREFERRED_HSIAO_72_64_VERSION'
	status = pthread_mutex_unlock(&(handle->PREFERRED_HSIAO_72_64_VERSION_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCSetPreferredHsiao_77_22_Version(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	return 0;
}

int clECCGetPreferredHsiao_77_22_Version(clECCHandle_t* handle, uint8_t* version) {
	if (handle == NULL || version == NULL) {
		return 1;
	}
	//check if handle is valid by dereferencing the handle pointer
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for 'PREFERRED_HSIAO_72_64_VERSION'
	status = pthread_mutex_lock(&(handle->PREFERRED_HSIAO_72_64_VERSION_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetPreferredHsiao_77_22_Version(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	*version = handle->PREFERRED_HSIAO_72_64_VERSION;

	//unlock mutex for 'PREFERRED_HSIAO_72_64_VERSION'
	status = pthread_mutex_unlock(&(handle->PREFERRED_HSIAO_72_64_VERSION_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetPreferredHsiao_77_22_Version(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	return 0;
}

int clECCSetMemoryScrubbingInterval(clECCHandle_t* handle, unsigned long long seconds) {
	if (handle == NULL) {
		return 1;
	}
	//check if handle is valid by dereferencing the handle pointer
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for 'EDAC_MEMORY_SCRUBBING_INTERVAL'
	status = pthread_mutex_lock(&(handle->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCSetMemoryScrubbingInterval(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	handle->EDAC_MEMORY_SCRUBBING_INTERVAL = seconds;

	//unlock mutex for 'EDAC_MEMORY_SCRUBBING_INTERVAL'
	status = pthread_mutex_unlock(&(handle->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCSetMemoryScrubbingInterval(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	return 0;
}

int clECCGetMemoryScrubbingInterval(clECCHandle_t* handle, unsigned long long* seconds) {
	if (handle == NULL || seconds == NULL) {
		return 1;
	}
	//check if handle is valid by dereferencing the handle pointer
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for 'EDAC_MEMORY_SCRUBBING_INTERVAL'
	status = pthread_mutex_lock(&(handle->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetMemoryScrubbingInterval(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	*seconds = handle->EDAC_MEMORY_SCRUBBING_INTERVAL;

	//unlock mutex for 'EDAC_MEMORY_SCRUBBING_INTERVAL'
	status = pthread_mutex_unlock(&(handle->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetMemoryScrubbingInterval(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return 4;
	}

	return 0;
}

//locking the EDAC mutex will block EDAC from occurring
int clECCLockEDACMutex(clECCHandle_t* handle) {
	if (handle == NULL) {
		return 1;
	}

	//make an attempt to lock the EDAC mutex
	int status = pthread_mutex_trylock(&(handle->EDAC_MUTEX));
	if (status == 0) {
		//EDAC mutex successfully locked
	}
	else if (status == EBUSY) {
		//EDAC mutex is already locked
	}
	else {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCLockEDACMutex(): pthread_mutex_trylock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
	
		//unable to obtain mutex lock for EDAC, force exit
		exit(1);
	}

	return 0;
}

//unlock EDAC mutex
int clECCUnlockEDACMutex(clECCHandle_t* handle) {
	if (handle == NULL) {
		return 1;
	}

	int status;

	//do not unlock mutex if mutex is not locked
	//otherwise, undefined behavior will occur
	status = pthread_mutex_trylock(&(handle->EDAC_MUTEX));
	if (status == 0) {
		//EDAC mutex successfully locked

		//unlock 'EDAC_MUTEX'
		status = pthread_mutex_unlock(&(handle->EDAC_MUTEX));
		if (status != 0) {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("clECCUnlockEDACMutex(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
	
			//unable to unlock mutex lock for EDAC, force exit
			exit(1);
		}
	}
	else if (status == EBUSY) {
		//EDAC mutex is already locked
		
		//unlock 'EDAC_MUTEX'
		status = pthread_mutex_unlock(&(handle->EDAC_MUTEX));
		if (status != 0) {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("clECCUnlockEDACMutex(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
			
			//unable to unlock mutex lock for EDAC, force exit
			exit(1);
		}
	}
	else {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCUnlockEDACMutex(): pthread_mutex_trylock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
	
		//unable to obtain mutex lock for EDAC, force exit
		exit(1);
	}

	return 0;
}

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int clECCGetTotalErrors(clECCHandle_t* handle, clECCMemObject_t* memory_object, uint64_t* errors, size_t errors_size) {
	if (handle == NULL || errors == NULL || memory_object == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;
	cl_event event;

	if (memory_object->total_errors_data_size > errors_size) {
		//memcpy() will go out of bounds if used on the given 'errors' argument
		return 1;
	}
	//lock memory object mutex
	status = pthread_mutex_lock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetTotalErrors(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to obtain mutex lock for memory object, gracefully return error
		return 4;
	}

	memcpy(errors, memory_object->total_errors, memory_object->total_errors_data_size);

	//unlock memory object mutex
	status = pthread_mutex_unlock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetTotalErrors(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to unlock mutex for memory object, force exit
		exit(1);
	}

	return 0;
}

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int clECCGetTotalErrorsWithCLMem(clECCHandle_t* handle, cl_mem device_memory, uint64_t* errors, size_t errors_size) {
	if (handle == NULL || errors == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for OpenCL memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetTotalErrorsWithCLMem(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to obtain mutex lock for memory object list, force exit
		exit(1);
	}

	clECCMemObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	clECCMemObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			if (current->data->total_errors_data_size > errors_size) {
				//memcpy() will go out of bounds if used on the given 'errors' argument
				return 1;
			}

			//lock corresponding memory object mutex
			status = pthread_mutex_lock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCGetTotalErrorsWithCLMem(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to obtain mutex lock for memory object, force exit
				exit(1);
			}

			memcpy(errors, current->data->total_errors, current->data->total_errors_data_size);

			//unlock corresponding memory object mutex
			status = pthread_mutex_unlock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCGetTotalErrorsWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to unlock mutex for memory object, force exit
				exit(1);
			}

			//unlock mutex for OpenCL memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCGetTotalErrorsWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to unlock mutex lock for memory object list, force exit
				exit(1);
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for OpenCL memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetTotalErrorsWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to unlock mutex lock for memory object list, force exit
		exit(1);
	}

	//the memory object could not be found in the given handle
	return 6;
}

int clECCGetTotalErrorsSize(clECCMemObject_t* memory_object, size_t* errors_size) {
	if (errors_size == NULL || memory_object == NULL) {
		return 1;
	}
	
	*errors_size = memory_object->total_errors_data_size;

	return 0;
}

int clECCGetTotalErrorsSizeWithCLMem(clECCHandle_t* handle, cl_mem device_memory, size_t* errors_size) {
	if (handle == NULL || errors_size == NULL) {
		return 1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for OpenCL memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetTotalErrorsSizeWithCLMem(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to obtain mutex lock for memory object list, force exit
		exit(1);
	}

	clECCMemObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	clECCMemObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			*errors_size = current->data->total_errors_data_size;

			//unlock mutex for OpenCL memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("clECCGetTotalErrorsSizeWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to unlock mutex lock for memory object list, force exit
				exit(1);
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for OpenCL memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("clECCGetTotalErrorsSizeWithCLMem(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to unlock mutex lock for memory object list, force exit
		exit(1);
	}

	//the memory object could not be found in the given handle
	return 6;
}

#ifdef __cplusplus
}
#endif
