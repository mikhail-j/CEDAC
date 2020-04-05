/* CUDA EDAC functions
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
#include "cuda_edac.h"
#include "cuda_edac_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

static void free_cuda_ecc_memory_object_list(char* string_buffer, cudaECCMemoryObjectList_t** objects) {
	if (objects == NULL) {
		return;
	}

	cudaECCMemoryObjectList_t* current = *objects;
	cudaECCMemoryObjectList_t* next;
	int status;

	while (current != NULL) {
		assert(cuMemFree(current->data->errors) == CUDA_SUCCESS);
		assert(cuMemFree(current->data->parity) == CUDA_SUCCESS);

		//unload CUDA module and set CUDA functions to NULL
		assert(cuModuleUnload(current->data->module) == CUDA_SUCCESS);
		current->data->edac_kernel = NULL;
		current->data->generator_kernel = NULL;

		//destroy corresponding memory object mutex
		status = pthread_mutex_destroy(&(current->data->mutex));
		if (status != 0) {
			while (status == EBUSY) {
				status = pthread_mutex_destroy(&(current->data->mutex));
			}
			if (status != 0 && status != EBUSY) {
				assert(strerror_r(status, string_buffer, 1024) == 0);
				printf("free_cuda_ecc_memory_object_list(): pthread_mutex_destroy(): error: %s\n", string_buffer);
				
				//encountered unrecoverable error, exit immediately
				exit(1);
			}
		}

		//free memory allocations
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

int cuECCInit(cudaECCHandle_t** handle) {
	if (handle == NULL) {
		return -1;
	}

	*handle = (cudaECCHandle_t *)malloc(sizeof(cudaECCHandle_t));
	if ((*handle) == NULL) {
		perror("malloc()");
		errno = 0;
		return -2;
	}

	(*handle)->MEMORY_ALLOCATIONS = (cudaECCMemoryObjectList_t **)malloc(sizeof(cudaECCMemoryObjectList_t *));
	if ((*handle)->MEMORY_ALLOCATIONS == NULL) {
		perror("malloc()");
		errno = 0;

		//free memory allocations
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return -2;
	}
	//initialize empty list
	*((*handle)->MEMORY_ALLOCATIONS) = NULL;

	int status = pthread_mutex_init(&((*handle)->MEMORY_ALLOCATIONS_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return -3;
	}

	status = pthread_mutex_init(&((*handle)->IS_ALIVE_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return -3;
	}

	status = pthread_cond_init(&((*handle)->EDAC_THREAD_WAIT_CONDITION), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCInit(): pthread_cond_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

		//destroy initialized mutexes
		assert(pthread_mutex_destroy(&((*handle)->IS_ALIVE_MUTEX)) == 0);
		assert(pthread_mutex_destroy(&((*handle)->MEMORY_ALLOCATIONS_MUTEX)) == 0);

		//free memory allocations
		free((*handle)->MEMORY_ALLOCATIONS);
		free(*handle);
		
		//set return value to NULL
		*handle = NULL;

		return -3;
	}

	status = pthread_mutex_init(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

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

		return -3;
	}


	status = pthread_mutex_init(&((*handle)->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

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

		return -3;
	}

	status = pthread_mutex_init(&((*handle)->PREFERRED_HSIAO_72_64_VERSION_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

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

		return -3;
	}

	status = pthread_mutex_init(&((*handle)->EDAC_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

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

		return -3;
	}

	status = pthread_mutex_init(&((*handle)->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX), NULL);
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

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
	status = pthread_create(&((*handle)->EDAC_THREAD), NULL, handle_cuda_edac_thread, (void *)(*handle));
	if (status != 0) {
		assert(strerror_r(status, (*handle)->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCInit(): pthread_mutex_init(): error: %s\n", (*handle)->ERRNO_STRING_BUFFER);

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

		return -3;
	}

	(*handle)->IS_ALIVE = true;

	//use Hsiao(72, 64) version 1 codes by default for 64-bit EDAC
	(*handle)->PREFERRED_HSIAO_72_64_VERSION = 1;

	//set memory scrubbing interval to 5 minutes
	(*handle)->EDAC_MEMORY_SCRUBBING_INTERVAL = 300;

	return 0;
}


//calling cuECCDestroy() on previous destroyed handle will produce unexpected behavior
void cuECCDestroy(cudaECCHandle_t* handle) {
	if (handle == NULL) {
		return;
	}

	int status;

	status = pthread_mutex_lock(&(handle->IS_ALIVE_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_mutex_lock(): %s\n", handle->ERRNO_STRING_BUFFER);
		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	handle->IS_ALIVE = false;

	status = pthread_mutex_unlock(&(handle->IS_ALIVE_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_mutex_unlock(): %s\n", handle->ERRNO_STRING_BUFFER);
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
				printf("cuECCDestroy(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
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
			while (t1 - t0 < 0.1) {
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
			printf("cuECCDestroy(): pthread_mutex_trylock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
			//unable to obtain mutex lock for wait condition, force exit
			exit(1);
		}
	}

	//wake up EDAC thread immediately
	status = pthread_cond_signal(&(handle->EDAC_THREAD_WAIT_CONDITION));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_cond_signal(): %s\n", handle->ERRNO_STRING_BUFFER);
		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//join thread first
	status = pthread_join(handle->EDAC_THREAD, NULL);
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//cuECCDestroy() maybe called without properly unlocking EDAC mutex beforehand
	status = pthread_mutex_destroy(&(handle->EDAC_MUTEX));
	if (status == 0) {}
	else if (status == EBUSY) {
		status = pthread_mutex_unlock(&(handle->EDAC_MUTEX));
		if (status != 0) {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("cuECCDestroy(): pthread_mutex_destroy(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);	
			//encountered unrecoverable error, exit immediately
			exit(1);
		}

		status = pthread_mutex_destroy(&(handle->EDAC_MUTEX));
		if (status != 0) {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("cuECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
			//encountered unrecoverable error, exit immediately
			exit(1);
		}
		//successfully destroyed EDAC mutex
	}
	else {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
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
		printf("cuECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->EDAC_THREAD_WAIT_CONDITION_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_cond_destroy(&(handle->EDAC_THREAD_WAIT_CONDITION));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_cond_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->IS_ALIVE_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	status = pthread_mutex_destroy(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCDestroy(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}


	//free all memory allocations in the linked list
	free_cuda_ecc_memory_object_list(handle->ERRNO_STRING_BUFFER, handle->MEMORY_ALLOCATIONS);

	//free memory allocations
	free(handle->MEMORY_ALLOCATIONS);
	free(handle);

	return;
}

//if 'memory_object' is NULL, the argument will be ignored
//int cuECCAddMemoryObject(cudaECCHandle_t* handle, int device_id, CUdeviceptr device_memory, cudaECCMemoryObject_t** memory_object) {
int cuECCAddMemoryObject(cudaECCHandle_t* handle, CUdeviceptr device_memory, cudaECCMemoryObject_t** memory_object) {
	if (handle == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	cudaECCMemoryObjectList_t* node;
	int status;

	//check if CUdeviceptr is already in handle
	node = *(handle->MEMORY_ALLOCATIONS);

	while (node != NULL) {
		if (node->data->data == device_memory) {
			//device pointer was previously added with cuECCAddMemoryObject()
			return -4;
		}

		//set next node
		node = node->next;
	}

	cudaECCMemoryObject_t* new_memory_object = (cudaECCMemoryObject_t *)malloc(sizeof(cudaECCMemoryObject_t));
	if (new_memory_object == NULL) {
		perror("malloc()");
		errno = 0;
		return -2;
	}

	node = (cudaECCMemoryObjectList_t *)malloc(sizeof(cudaECCMemoryObjectList_t));
	if (node == NULL) {
		perror("malloc()");
		errno = 0;

		//free memory allocations
		free(new_memory_object);

		return -2;
	}
	node->data = new_memory_object;
	node->next = *(handle->MEMORY_ALLOCATIONS);

	new_memory_object->total_errors_data_size = 2 * sizeof(uint64_t);

	//assign arguments in new_memory_object
	new_memory_object->data = device_memory;

	status = cuPointerGetAttribute(&(new_memory_object->context), CU_POINTER_ATTRIBUTE_CONTEXT, new_memory_object->data);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuPointerGetAttribute(): %s\n", error_string);

		//free memory allocations
		free(node);
		free(new_memory_object);

		return status;
	}

	int device_id;
	status = cuPointerGetAttribute(&device_id, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, new_memory_object->data);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuPointerGetAttribute(): %s\n", error_string);

		//free memory allocations
		free(node);
		free(new_memory_object);

		return status;
	}

	status = cuDeviceGet(&(new_memory_object->device), device_id);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuDeviceGet(): %s\n", error_string);

		//free memory allocations
		free(node);
		free(new_memory_object);

		return status;
	}

	status = cuDeviceGetAttribute(&(new_memory_object->DEVICE_MULTIPROCESSOR_COUNT), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, new_memory_object->device);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuDeviceGetAttribute(): %s\n", error_string);

		//free memory allocations
		free(node);
		free(new_memory_object);

		return status;
	}

	status = cuDeviceGetAttribute(&(new_memory_object->DEVICE_MAX_THREADS_PER_BLOCK), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, new_memory_object->device);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuDeviceGetAttribute(): %s\n", error_string);

		//free memory allocations
		free(node);
		free(new_memory_object);

		return status;
	}

	status = cuCtxSetCurrent(new_memory_object->context);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuCtxSetCurrent(): %s\n", error_string);

		//free memory allocations
		free(node);
		free(new_memory_object);

		return status;
	}

	size_t device_memory_size;

	status = cuMemGetAddressRange(&(new_memory_object->data), &device_memory_size, device_memory);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuMemGetAddressRange(): %s\n", error_string);

		//free memory allocations
		free(node);
		free(new_memory_object);

		return status;
	}
	assert(new_memory_object->data == device_memory);

	//ECC only implemented for 16-bit, 32-bit, and 64-bit data types
	if ((device_memory_size % sizeof(uint64_t)) == 0) {
		new_memory_object->element_count = device_memory_size / sizeof(uint64_t);
		new_memory_object->element_size = sizeof(uint64_t);
		unsigned char hsiao_72_64_version;

		//throw assertion error if mutex is not usable
		assert(cuECCGetPreferredHsiao_77_22_Version(handle, &hsiao_72_64_version) == 0);

		if (hsiao_72_64_version == 1) {
			status = cuModuleLoadData(&(new_memory_object->module), hsiao_72_64_v1_cuda);
			if (status != CUDA_SUCCESS) {
				const char * error_string;
				assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
				printf("cuModuleLoadData(): %s\n", error_string);
		
				//free memory allocations
				free(node);
				free(new_memory_object);
		
				return status;
			}
			status = cuModuleGetFunction(&(new_memory_object->generator_kernel), new_memory_object->module, "generate_parity_hsiao_72_64_v1");
			if (status != CUDA_SUCCESS) {
				const char * error_string;
				assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
				printf("cuModuleGetFunction(): %s\n", error_string);
				
				//unload CUmodule
				assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object);
		
				return status;
			}
			status = cuModuleGetFunction(&(new_memory_object->edac_kernel), new_memory_object->module, "edac_hsiao_72_64_v1");
			if (status != CUDA_SUCCESS) {
				const char * error_string;
				assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
				printf("cuModuleGetFunction(): %s\n", error_string);
				
				//unload CUmodule
				assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
				
				//free memory allocations
				free(node);
				free(new_memory_object);
		
				return status;
			}
		}
		else if (hsiao_72_64_version == 2) {
			status = cuModuleLoadData(&(new_memory_object->module), hsiao_72_64_v2_cuda);
			if (status != CUDA_SUCCESS) {
				const char * error_string;
				assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
				printf("cuModuleLoadData(): %s\n", error_string);
		
				//free memory allocations
				free(node);
				free(new_memory_object);
		
				return status;
			}
			status = cuModuleGetFunction(&(new_memory_object->generator_kernel), new_memory_object->module, "generate_parity_hsiao_72_64_v2");
			if (status != CUDA_SUCCESS) {
				const char * error_string;
				assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
				printf("cuModuleGetFunction(): %s\n", error_string);
				
				//unload CUmodule
				assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);

				//free memory allocations
				free(node);
				free(new_memory_object);
		
				return status;
			}
			status = cuModuleGetFunction(&(new_memory_object->edac_kernel), new_memory_object->module, "edac_hsiao_72_64_v2");
			if (status != CUDA_SUCCESS) {
				const char * error_string;
				assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
				printf("cuModuleGetFunction(): %s\n", error_string);
				
				//unload CUmodule
				assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
				
				//free memory allocations
				free(node);
				free(new_memory_object);
		
				return status;
			}
		}	
		else {
			printf("cuECCAddMemoryObject(): error: found invalid 'PREFERRED_HSIAO_72_64_VERSION' value, %d!\n", hsiao_72_64_version);

			//free memory allocations
			free(node);
			free(new_memory_object);

			return -6;
		}
	}
	else if ((device_memory_size % sizeof(uint32_t)) == 0) {
		new_memory_object->element_count = device_memory_size / sizeof(uint32_t);
		new_memory_object->element_size = sizeof(uint32_t);

		status = cuModuleLoadData(&(new_memory_object->module), hsiao_39_32_cuda);
		if (status != CUDA_SUCCESS) {
			const char * error_string;
			assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
			printf("cuModuleLoadData(): %s\n", error_string);
		
			//free memory allocations
			free(node);
			free(new_memory_object);
		
			return status;
		}
		status = cuModuleGetFunction(&(new_memory_object->generator_kernel), new_memory_object->module, "generate_parity_hsiao_39_32");
		if (status != CUDA_SUCCESS) {
			const char * error_string;
			assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
			printf("cuModuleGetFunction(): %s\n", error_string);
			
			//unload CUmodule
			assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object);
		
			return status;
		}
		status = cuModuleGetFunction(&(new_memory_object->edac_kernel), new_memory_object->module, "edac_hsiao_39_32");
		if (status != CUDA_SUCCESS) {
			const char * error_string;
			assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
			printf("cuModuleGetFunction(): %s\n", error_string);
			
			//unload CUmodule
			assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
			
			//free memory allocations
			free(node);
			free(new_memory_object);
		
			return status;
		}
	}
	else if ((device_memory_size % sizeof(uint16_t)) == 0) {
		new_memory_object->element_count = device_memory_size / sizeof(uint16_t);
		new_memory_object->element_size = sizeof(uint16_t);

		status = cuModuleLoadData(&(new_memory_object->module), hsiao_22_16_cuda);
		if (status != CUDA_SUCCESS) {
			const char * error_string;
			assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
			printf("cuModuleLoadData(): %s\n", error_string);
		
			//free memory allocations
			free(node);
			free(new_memory_object);
		
			return status;
		}
		status = cuModuleGetFunction(&(new_memory_object->generator_kernel), new_memory_object->module, "generate_parity_hsiao_22_16");
		if (status != CUDA_SUCCESS) {
			const char * error_string;
			assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
			printf("cuModuleGetFunction(): %s\n", error_string);
			
			//unload CUmodule
			assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);

			//free memory allocations
			free(node);
			free(new_memory_object);
		
			return status;
		}
		status = cuModuleGetFunction(&(new_memory_object->edac_kernel), new_memory_object->module, "edac_hsiao_22_16");
		if (status != CUDA_SUCCESS) {
			const char * error_string;
			assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
			printf("cuModuleGetFunction(): %s\n", error_string);
			
			//unload CUmodule
			assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
			
			//free memory allocations
			free(node);
			free(new_memory_object);
		
			return status;
		}
	}
	else {
		//free memory allocations
		free(node);
		free(new_memory_object);

		printf("cuECCAddMemoryObject(): error: encountered unsupported device memory size, %zu!\n", device_memory_size);
		return status;
	}

	//allocate parity bits in device memory
	status = cuMemAlloc(&(new_memory_object->parity), sizeof(uint8_t) * new_memory_object->element_count);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuMemAlloc(): %s\n", error_string);
		
		//unload CUmodule
		assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
		
		//free memory allocations
		free(node);
		free(new_memory_object);
	
		return status;
	}

	//allocate error counter in device memory
	status = cuMemAlloc(&(new_memory_object->errors), sizeof(uint8_t) * new_memory_object->element_count);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuMemAlloc(): %s\n", error_string);
		
		//free device memory allocations
		assert(cuMemFree(new_memory_object->parity) == CUDA_SUCCESS);

		//unload CUmodule
		assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
		
		//free memory allocations
		free(node);
		free(new_memory_object);
	
		return status;
	}

	//initialize error count variables (host memory) to 0
	new_memory_object->total_errors[0] = 0;
	new_memory_object->total_errors[1] = 0;

	//copy data from host to device
	status = cuMemcpyHtoD(new_memory_object->errors, new_memory_object->total_errors, sizeof(uint64_t) * 2);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuMemcpyHtoD(): %s\n", error_string);
		
		//free device memory allocations
		assert(cuMemFree(new_memory_object->errors) == CUDA_SUCCESS);
		assert(cuMemFree(new_memory_object->parity) == CUDA_SUCCESS);

		//unload CUmodule
		assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
		
		//free memory allocations
		free(node);
		free(new_memory_object);
	
		return status;
	}

	//set CUDA kernel arguments
	new_memory_object->kernel_arguments[0] = &(new_memory_object->data);
	new_memory_object->kernel_arguments[1] = &(new_memory_object->parity);
	new_memory_object->kernel_arguments[2] = &(new_memory_object->element_count);
	new_memory_object->kernel_arguments[3] = &(new_memory_object->errors);

	//use generator kernel to generate parity bits
	status = cuLaunchKernel(new_memory_object->generator_kernel,
						new_memory_object->DEVICE_MULTIPROCESSOR_COUNT, 1, 1,
						new_memory_object->DEVICE_MAX_THREADS_PER_BLOCK, 1, 1,
						0,
						NULL,	//use the zero CUDA stream
						new_memory_object->kernel_arguments,
						NULL);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuLaunchKernel(): %s\n", error_string);
		
		//free device memory allocations
		assert(cuMemFree(new_memory_object->errors) == CUDA_SUCCESS);
		assert(cuMemFree(new_memory_object->parity) == CUDA_SUCCESS);

		//unload CUmodule
		assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
		
		//free memory allocations
		free(node);
		free(new_memory_object);
	
		return status;
	}

	//wait for CUDA kernel to finish
	status = cuStreamSynchronize(NULL);
	if (status != CUDA_SUCCESS) {
		const char * error_string;
		assert(cuGetErrorString(status, &error_string) == CUDA_SUCCESS);
		printf("cuStreamSynchronize(): %s\n", error_string);
		
		//free device memory allocations
		assert(cuMemFree(new_memory_object->errors) == CUDA_SUCCESS);
		assert(cuMemFree(new_memory_object->parity) == CUDA_SUCCESS);

		//unload CUmodule
		assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
		
		//free memory allocations
		free(node);
		free(new_memory_object);
	
		return status;
	}

	//initialize memory object mutex
	status = pthread_mutex_init(&(new_memory_object->mutex), NULL);
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCAddMemoryObject(): pthread_mutex_init(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//free device memory allocations
		assert(cuMemFree(new_memory_object->errors) == CUDA_SUCCESS);
		assert(cuMemFree(new_memory_object->parity) == CUDA_SUCCESS);

		//unload CUmodule
		assert(cuModuleUnload(new_memory_object->module) == CUDA_SUCCESS);
		
		//free memory allocations
		free(node);
		free(new_memory_object);

		return -3;
	}

	//push 'node' into linked list
	*(handle->MEMORY_ALLOCATIONS) = node;

	if (memory_object != NULL) {
		*memory_object = new_memory_object;
	}

	return 0;
}


//cuECCRemoveMemoryObject() will call free() on 'memory_object' on success
int cuECCRemoveMemoryObject(cudaECCHandle_t* handle, cudaECCMemoryObject_t* memory_object) {
	if (handle == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//lock mutex for CUDA memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCRemoveMemoryObject(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	cudaECCMemoryObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	cudaECCMemoryObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data == memory_object) {
			//free device memory allocations
			assert(cuMemFree(current->data->errors) == CUDA_SUCCESS);
			assert(cuMemFree(current->data->parity) == CUDA_SUCCESS);

			//unload CUmodule
			assert(cuModuleUnload(current->data->module) == CUDA_SUCCESS);

			//destroy corresponding memory object mutex
			status = pthread_mutex_destroy(&(current->data->mutex));
			if (status != 0) {
				while (status == EBUSY) {
					//retry pthread_mutex_destroy() on the mutex if another process was using the mutex
					status = pthread_mutex_destroy(&(current->data->mutex));
				}
				if (status != 0 && status != EBUSY) {
					assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
					printf("cuECCRemoveMemoryObject(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
					
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
			free(current->data);
			free(current);

			//unlock mutex for CUDA memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCRemoveMemoryObject(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//encountered unrecoverable error, exit immediately
				exit(1);
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for CUDA memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCRemoveMemoryObject(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//the memory object could not be found in the given handle
	return -5;
}

int cuECCRemoveMemoryObjectWithDevicePointer(cudaECCHandle_t* handle, CUdeviceptr device_memory) {
	if (handle == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//lock mutex for OpenCL memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCRemoveMemoryObjectWithDevicePointer(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	cudaECCMemoryObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	cudaECCMemoryObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			//free device memory allocations
			assert(cuMemFree(current->data->errors) == CUDA_SUCCESS);
			assert(cuMemFree(current->data->parity) == CUDA_SUCCESS);

			//unload CUmodule
			assert(cuModuleUnload(current->data->module) == CUDA_SUCCESS);

			//destroy corresponding memory object mutex
			status = pthread_mutex_destroy(&(current->data->mutex));
			if (status != 0) {
				while (status == EBUSY) {
					//retry pthread_mutex_destroy() on the mutex if another process was using the mutex
					status = pthread_mutex_destroy(&(current->data->mutex));
				}
				if (status != 0 && status != EBUSY) {
					assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
					printf("cuECCRemoveMemoryObjectWithDevicePointer(): pthread_mutex_destroy(): error: %s\n", handle->ERRNO_STRING_BUFFER);
					
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
			free(current->data);
			free(current);

			//unlock mutex for OpenCL memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCRemoveMemoryObjectWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
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
		printf("cuECCRemoveMemoryObjectWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//the memory object could not be found in the given handle
	return -5;
}

//cuECCUpdateMemoryObject() accepts 'cudaECCMemoryObject_t *' instead of 'CUdeviceptr', avoiding
//the lookup of CUDA memory object.
//Note: Using cuECCUpdateMemoryObject() without locking EDAC mutex creates a race condition.
int cuECCUpdateMemoryObject(cudaECCHandle_t* handle, cudaECCMemoryObject_t* memory_object) {
	if (handle == NULL || memory_object == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//set current CUDA context
	status = cuCtxSetCurrent(memory_object->context);
	if (status != CUDA_SUCCESS) {
		return status;
	}

	//use generator kernel to generate parity bits
	status = cuLaunchKernel(memory_object->generator_kernel,
						memory_object->DEVICE_MULTIPROCESSOR_COUNT, 1, 1,
						memory_object->DEVICE_MAX_THREADS_PER_BLOCK, 1, 1,
						0,
						NULL,	//use the zero CUDA stream
						memory_object->kernel_arguments,
						NULL);
	if (status != CUDA_SUCCESS) {
		return status;
	}

	//wait for CUDA kernel to finish
	status = cuStreamSynchronize(NULL);

    //immediately return CUDA error for cuStreamSynchronize()
	return status;
}

//update CUDA memory object ECC
int cuECCUpdateMemoryObjectWithDevicePointer(cudaECCHandle_t* handle, CUdeviceptr device_memory) {
	if (handle == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//lock mutex for OpenCL memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCUpdateMemoryObjectWithDevicePointer(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	cudaECCMemoryObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	cudaECCMemoryObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			//set current CUDA context
			status = cuCtxSetCurrent(current->data->context);
			if (status != CUDA_SUCCESS) {
				//unlock mutex for CUDA memory allocations
				int mutex_status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
				if (mutex_status != 0) {
					assert(strerror_r(mutex_status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
					printf("cuECCUpdateMemoryObjectWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
					
					//encountered unrecoverable error, exit immediately
					exit(1);
				}

				//return CUDA error
				return status;
			}

			//use generator kernel to generate parity bits
			status = cuLaunchKernel(current->data->generator_kernel,
								current->data->DEVICE_MULTIPROCESSOR_COUNT, 1, 1,
								current->data->DEVICE_MAX_THREADS_PER_BLOCK, 1, 1,
								0,
								NULL,	//use the zero CUDA stream
								current->data->kernel_arguments,
								NULL);
			if (status != CUDA_SUCCESS) {
				//unlock mutex for CUDA memory allocations
				int mutex_status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
				if (mutex_status != 0) {
					assert(strerror_r(mutex_status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
					printf("cuECCUpdateMemoryObjectWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
					
					//encountered unrecoverable error, exit immediately
					exit(1);
				}

				//return CUDA error
				return status;
			}

			//wait for CUDA kernel to finish
			status = cuStreamSynchronize(NULL);
			if (status != CUDA_SUCCESS) {
				//unlock mutex for CUDA memory allocations
				int mutex_status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
				if (mutex_status != 0) {
					assert(strerror_r(mutex_status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
					printf("cuECCUpdateMemoryObjectWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
					
					//encountered unrecoverable error, exit immediately
					exit(1);
				}

				//return CUDA error
				return status;
			}

			//unlock mutex for CUDA memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCUpdateMemoryObjectWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//encountered unrecoverable error, exit immediately
				exit(1);
			}

			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for CUDA memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCUpdateMemoryObjectWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//the memory object could not be found in the given handle
	return -5;
}

//obtain parity bits returned as a device memory allocation
int cuECCGetMemoryObjectParityBits(cudaECCHandle_t* handle, cudaECCMemoryObject_t* memory_object, CUdeviceptr* parity_memory) {
	if (handle == NULL
		|| memory_object == NULL
		|| parity_memory == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//lock memory object mutex
	status = pthread_mutex_lock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetMemoryObjectParityBits(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to obtain mutex lock for memory object, force exit
		exit(1);
	}

	//assign associated parity bits OpenCL memory to dereferenced 'parity_memory' pointer
	*parity_memory = memory_object->parity;

	//unlock memory object mutex
	status = pthread_mutex_unlock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetMemoryObjectParityBits(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to unlock mutex for memory object, force exit
		exit(1);
	}

	//the memory object could not be found in the given handle
	return 0;
}

//obtain parity bits returned as a device memory allocation
int cuECCGetMemoryObjectParityBitsWithDevicePointer(cudaECCHandle_t* handle, CUdeviceptr device_memory, CUdeviceptr* parity_memory) {
	if (handle == NULL || parity_memory == NULL) {
		return -1;
	}
	//check if handle is valid by dereferencing the handle pointer
	if (!(handle->IS_ALIVE)) {
		return 1;
	}

	int status;

	//lock mutex for CUDA memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetMemoryObjectParityBitsWithDevicePointer(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	cudaECCMemoryObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	cudaECCMemoryObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			//lock corresponding memory object mutex
			status = pthread_mutex_lock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCGetMemoryObjectParityBitsWithDevicePointer(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to obtain mutex lock for memory object, force exit
				exit(1);
			}

			//assign associated parity bits CUDA device memory to dereferenced 'parity_memory' pointer
			*parity_memory = current->data->parity;
			
			//unlock corresponding memory object mutex
			status = pthread_mutex_unlock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCGetMemoryObjectParityBitsWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to unlock mutex for memory object, force exit
				exit(1);
			}

			//unlock mutex for CUDA memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCGetMemoryObjectParityBitsWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//encountered unrecoverable error, exit immediately
				exit(1);
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for CUDA memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetMemoryObjectParityBitsWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//encountered unrecoverable error, exit immediately
		exit(1);
	}

	//the memory object could not be found in the given handle
	return -5;
}

int cuECCSetPreferredHsiao_77_22_Version(cudaECCHandle_t* handle, unsigned char version) {
	if (handle == NULL || (version != 1 && version != 2)) {
		return -1;
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
		printf("cuECCSetPreferredHsiao_77_22_Version(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return -3;
	}

	handle->PREFERRED_HSIAO_72_64_VERSION = version;

	//unlock mutex for 'PREFERRED_HSIAO_72_64_VERSION'
	status = pthread_mutex_unlock(&(handle->PREFERRED_HSIAO_72_64_VERSION_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCSetPreferredHsiao_77_22_Version(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return -3;
	}

	return 0;
}

int cuECCGetPreferredHsiao_77_22_Version(cudaECCHandle_t* handle, unsigned char* version) {
	if (handle == NULL || version == NULL) {
		return -1;
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
		printf("cuECCGetPreferredHsiao_77_22_Version(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return -3;
	}

	*version = handle->PREFERRED_HSIAO_72_64_VERSION;

	//unlock mutex for 'PREFERRED_HSIAO_72_64_VERSION'
	status = pthread_mutex_unlock(&(handle->PREFERRED_HSIAO_72_64_VERSION_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetPreferredHsiao_77_22_Version(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return -3;
	}

	return 0;
}

int cuECCSetMemoryScrubbingInterval(cudaECCHandle_t* handle, unsigned long long seconds) {
	if (handle == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//lock mutex for 'EDAC_MEMORY_SCRUBBING_INTERVAL'
	status = pthread_mutex_lock(&(handle->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCSetMemoryScrubbingInterval(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return -3;
	}

	handle->EDAC_MEMORY_SCRUBBING_INTERVAL = seconds;

	//unlock mutex for 'EDAC_MEMORY_SCRUBBING_INTERVAL'
	status = pthread_mutex_unlock(&(handle->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCSetMemoryScrubbingInterval(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return -3;
	}

	return 0;
}

int cuECCGetMemoryScrubbingInterval(cudaECCHandle_t* handle, unsigned long long* seconds) {
	if (handle == NULL || seconds == NULL || !(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//lock mutex for 'EDAC_MEMORY_SCRUBBING_INTERVAL'
	status = pthread_mutex_lock(&(handle->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetMemoryScrubbingInterval(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return -3;
	}

	*seconds = handle->EDAC_MEMORY_SCRUBBING_INTERVAL;

	//unlock mutex for 'EDAC_MEMORY_SCRUBBING_INTERVAL'
	status = pthread_mutex_unlock(&(handle->EDAC_MEMORY_SCRUBBING_INTERVAL_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetMemoryScrubbingInterval(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		return -3;
	}

	return 0;
}

//locking the EDAC mutex will block EDAC from occurring
int cuECCLockEDACMutex(cudaECCHandle_t* handle) {
	if (handle == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
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
		printf("cuECCLockEDACMutex(): pthread_mutex_trylock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
	
		//unable to obtain mutex lock for EDAC, force exit
		exit(1);
	}

	return 0;
}


//unlock EDAC mutex
int cuECCUnlockEDACMutex(cudaECCHandle_t* handle) {
	if (handle == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
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
			printf("cuECCUnlockEDACMutex(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
	
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
			printf("cuECCUnlockEDACMutex(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
			
			//unable to unlock mutex lock for EDAC, force exit
			exit(1);
		}
	}
	else {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCUnlockEDACMutex(): pthread_mutex_trylock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
	
		//unable to obtain mutex lock for EDAC, force exit
		exit(1);
	}

	return 0;
}

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int cuECCGetTotalErrors(cudaECCHandle_t* handle, cudaECCMemoryObject_t* memory_object, uint64_t* errors, size_t errors_size) {
	if (handle == NULL || errors == NULL || memory_object == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	if (memory_object->total_errors_data_size > errors_size) {
		//memcpy() will go out of bounds if used on the given 'errors' argument
		return -1;
	}
	//lock memory object mutex
	status = pthread_mutex_lock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetTotalErrors(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to lock mutex for memory object, gracefully return error
		return -3;
	}

	memcpy(errors, memory_object->total_errors, memory_object->total_errors_data_size);

	//unlock memory object mutex
	status = pthread_mutex_unlock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetTotalErrors(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to unlock mutex for memory object, force exit
		exit(1);
	}

	return 0;
}

//errors[0] = number of single bit errors detected
//errors[1] = number of double bit errors detected
int cuECCGetTotalErrorsWithDevicePointer(cudaECCHandle_t* handle, CUdeviceptr device_memory, uint64_t* errors, size_t errors_size) {
	if (handle == NULL || errors == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//lock mutex for CUDA memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetTotalErrorsWithDevicePointer(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to obtain mutex lock for memory object list, force exit
		exit(1);
	}

	cudaECCMemoryObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	cudaECCMemoryObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			if (current->data->total_errors_data_size > errors_size) {
				//memcpy() will go out of bounds if used on the given 'errors' argument
				return -1;
			}

			//lock corresponding memory object mutex
			status = pthread_mutex_lock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCGetTotalErrorsWithDevicePointer(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to obtain mutex lock for memory object, force exit
				exit(1);
			}

			memcpy(errors, current->data->total_errors, current->data->total_errors_data_size);

			//unlock corresponding memory object mutex
			status = pthread_mutex_unlock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCGetTotalErrorsWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to unlock mutex for memory object, force exit
				exit(1);
			}

			//unlock mutex for CUDA memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCGetTotalErrorsWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to unlock mutex lock for memory object list, force exit
				exit(1);
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for CUDA memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetTotalErrorsWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to unlock mutex lock for memory object list, force exit
		exit(1);
	}

	//the memory object could not be found in the given handle
	return -5;
}

int cuECCGetTotalErrorsSize(cudaECCMemoryObject_t* memory_object, size_t* errors_size) {
	if (errors_size == NULL || memory_object == NULL) {
		return -1;
	}

	*errors_size = memory_object->total_errors_data_size;

	return 0;
}

int cuECCGetTotalErrorsSizeWithDevicePointer(cudaECCHandle_t* handle, CUdeviceptr device_memory, size_t* errors_size) {
	if (handle == NULL || errors_size == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//lock mutex for CUDA memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetTotalErrorsSizeWithDevicePointer(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to obtain mutex lock for memory object list, force exit
		exit(1);
	}

	cudaECCMemoryObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);
	cudaECCMemoryObjectList_t* previous = NULL;

	while (current != NULL) {
		if (current->data->data == device_memory) {
			*errors_size = current->data->total_errors_data_size;

			//unlock mutex for CUDA memory allocations
			status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("cuECCGetTotalErrorsSizeWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to unlock mutex lock for memory object list, force exit
				exit(1);
			}
			return 0;
		}

		//set next node
		previous = current;
		current = current->next;
	}

	//unlock mutex for CUDA memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCGetTotalErrorsSizeWithDevicePointer(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to unlock mutex lock for memory object list, force exit
		exit(1);
	}

	//the memory object could not be found in the given handle
	return -5;
}

int cuECCEDAC(cudaECCHandle_t* handle, cudaECCMemoryObject_t* memory_object) {
	if (handle == NULL || memory_object == NULL) {
		return -1;
	}
	//check if handle is valid
	if (!(handle->IS_ALIVE)) {
		return -1;
	}

	int status;

	//lock memory object mutex
	status = pthread_mutex_lock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCEDAC(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to obtain mutex lock for memory object, gracefully return error
		return -3;
	}

	//do EDAC only if double bit errors have not occurred
	if (memory_object->total_errors[1] != 0) {
		//unlock memory object mutex
		status = pthread_mutex_unlock(&(memory_object->mutex));
		if (status != 0) {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("cuECCEDAC(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
			
			//unable to unlock mutex for memory object, force exit
			exit(1);
		}

		//uncorrectable error 
		return -7;
	}

	//set current CUDA context
	assert(cuCtxSetCurrent(memory_object->context) == CUDA_SUCCESS);

	//use EDAC kernel to perform error detection and correction
	assert(cuLaunchKernel(memory_object->edac_kernel,
						memory_object->DEVICE_MULTIPROCESSOR_COUNT, 1, 1,
						memory_object->DEVICE_MAX_THREADS_PER_BLOCK, 1, 1,
						0,
						NULL,	//use the zero CUDA stream
						memory_object->kernel_arguments,
						NULL) == CUDA_SUCCESS);

	//wait for CUDA kernel to finish
	assert(cuStreamSynchronize(NULL) == CUDA_SUCCESS);

	//copy error count from device to host
	assert(cuMemcpyDtoH(memory_object->total_errors, memory_object->errors, memory_object->total_errors_data_size) == CUDA_SUCCESS);

	//unlock memory object mutex
	status = pthread_mutex_unlock(&(memory_object->mutex));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("cuECCEDAC(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
		//unable to unlock mutex for memory object, force exit
		exit(1);
	}

	return 0;
}

#ifdef __cplusplus
}
#endif
