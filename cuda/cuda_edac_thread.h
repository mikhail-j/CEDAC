/* CUDA EDAC thread function definitions
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
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <time.h>

#ifndef CUDA_ECC_EDAC_THREAD_H
#define CUDA_ECC_EDAC_THREAD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_edac.h"

int is_cuda_edac_thread_alive(cudaECCHandle_t* handle, bool* retval);

__attribute__ ((visibility ("hidden"))) int memory_scrub_cuda_memory_allocations(cudaECCHandle_t* handle);

__attribute__ ((visibility ("hidden"))) void * handle_cuda_edac_thread(void * args);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_ECC_EDAC_THREAD_H */
