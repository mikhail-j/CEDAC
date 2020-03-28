/* OpenCL EDAC thread functions
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
#include "opencl_edac_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

int is_opencl_edac_thread_alive(clECCHandle_t * handle, bool * retval) {
	if ((handle == NULL) || (retval == NULL)) {
		return 1;
	}

	int status;

	status = pthread_mutex_lock(&(handle->IS_ALIVE_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("is_opencl_edac_thread_alive(): pthread_mutex_lock(): %s\n", handle->ERRNO_STRING_BUFFER);
		return 4;
	}

	*retval = handle->IS_ALIVE;

	status = pthread_mutex_unlock(&(handle->IS_ALIVE_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("is_opencl_edac_thread_alive(): pthread_mutex_unlock(): %s\n", handle->ERRNO_STRING_BUFFER);
		return 4;
	}

	return 0;
}

__attribute__ ((visibility ("hidden"))) int memory_scrub_opencl_memory_allocations(clECCHandle_t* handle) {
	if (handle == NULL) {
		return 1;
	}

	int status;
	cl_event event;

	//lock mutex for OpenCL memory allocations
	status = pthread_mutex_lock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("memory_scrub_opencl_memory_allocations(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to obtain mutex lock for memory object list, force exit
		exit(1);
	}

	clECCMemObjectList_t* current = *(handle->MEMORY_ALLOCATIONS);

	while (current != NULL) {
		//do EDAC only if double bit errors have not occurred
		if (current->data->total_errors[1] == 0) {
			//lock corresponding memory object mutex
			status = pthread_mutex_lock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("memory_scrub_opencl_memory_allocations(): pthread_mutex_lock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to obtain mutex lock for memory object, force exit
				exit(1);
			}

			//use EDAC kernel to perform error detection and correction
			assert(clEnqueueNDRangeKernel(current->data->queue,
    		                            current->data->edac_kernel,
    		                            1,
    		                            NULL,
    		                            &(current->data->KERNEL_GLOBAL_WORK_SIZE),
    		                            &(current->data->DEVICE_MAX_WORK_GROUP_SIZE),
    		                            0,
    		                            NULL,
    		                            &event) == CL_SUCCESS);
    		assert(clWaitForEvents(1, &event) == CL_SUCCESS);

    		//copy error count from device to host
    		assert(clEnqueueReadBuffer(current->data->queue, current->data->errors, CL_TRUE, 0, 2 * sizeof(uint64_t), current->data->total_errors, 0, NULL, &event) == CL_SUCCESS);
    		assert(clWaitForEvents(1, &event) == CL_SUCCESS);

			//unlock corresponding memory object mutex
			status = pthread_mutex_unlock(&(current->data->mutex));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("memory_scrub_opencl_memory_allocations(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
				
				//unable to unlock mutex lock for memory object, force exit
				exit(1);
			}
    	}

		current = current->next;
	}

	//unlock mutex for OpenCL memory allocations
	status = pthread_mutex_unlock(&(handle->MEMORY_ALLOCATIONS_MUTEX));
	if (status != 0) {
		assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
		printf("memory_scrub_opencl_memory_allocations(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);

		//unable to unlock mutex lock for memory object list, force exit
		exit(1);
	}

	return 0;
}

__attribute__ ((visibility ("hidden"))) void * handle_opencl_edac_thread(void * args) {
	clECCHandle_t* handle = (clECCHandle_t *)args;

	bool is_alive = true;
	int status;
	struct timespec wakeup_time;
	unsigned int relative_time = 0;
	unsigned long long delay;

	while (is_alive) {
		status = pthread_mutex_trylock(&(handle->EDAC_MUTEX));
		if (status == 0) {
			//EDAC mutex successfully locked
			assert(memory_scrub_opencl_memory_allocations(handle) == 0);

			//unlock mutex for EDAC
			status = pthread_mutex_unlock(&(handle->EDAC_MUTEX));
			if (status != 0) {
				assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
				printf("handle_opencl_edac_thread(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
				//unable to unlock mutex lock for EDAC, force exit
				exit(1);
			}
		}
		else if (status == EBUSY) {
			//EDAC mutex is already locked, do nothing
		}
		else {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("handle_opencl_edac_thread(): pthread_mutex_unlock(): error: %s\n", handle->ERRNO_STRING_BUFFER);
		
			//unable to obtain mutex lock for EDAC, force exit
			exit(1);
		}

		status = clock_gettime(CLOCK_REALTIME, &wakeup_time);
		if (status != 0) {
			perror("clock_gettime()");
			errno = 0;

			//unable to obtain time for delay, force exit
			exit(1);
		}
		//run memory scrubbing in intervals if thread is still alive
		assert(clECCGetMemoryScrubbingInterval(handle, &delay) == 0);
		wakeup_time.tv_sec = wakeup_time.tv_sec + delay;

		//unlock wait condition mutex mutex
		status = pthread_mutex_lock(&(handle->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX));
		if (status != 0) {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("handle_opencl_edac_thread(): pthread_mutex_lock(): %s\n", handle->ERRNO_STRING_BUFFER);
			//encountered unrecoverable error, exit immediately
			exit(1);
		}
		status = pthread_cond_timedwait(&(handle->EDAC_THREAD_WAIT_CONDITION),
									&(handle->EDAC_THREAD_WAIT_CONDITION_MUTEX),
									&wakeup_time);
		if (status == 0) {
			//do nothing
		}
		else if (status == ETIMEDOUT) {
			//initiate memory scrubbing
		}
		else {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("handle_opencl_edac_thread(): pthread_cond_timedwait(): %s\n", handle->ERRNO_STRING_BUFFER);

			//unrecoverable error? force exit
			exit(1);
		}
		//unlock wait condition mutex mutex
		status = pthread_mutex_unlock(&(handle->EDAC_THREAD_WAIT_CONDITION_MUTEX_MUTEX));
		if (status != 0) {
			assert(strerror_r(status, handle->ERRNO_STRING_BUFFER, 1024) == 0);
			printf("handle_opencl_edac_thread(): pthread_mutex_unlock(): %s\n", handle->ERRNO_STRING_BUFFER);
			//encountered unrecoverable error, exit immediately
			exit(1);
		}

		//lock 'IS_ALIVE' mutex and check the value of 'IS_ALIVE'
		assert(is_opencl_edac_thread_alive(handle, &is_alive) == 0);
	}

	pthread_exit(NULL);
}


#ifdef __cplusplus
}
#endif
