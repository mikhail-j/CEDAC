/* CUDA EDAC assertion tests
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
#include <cuda.h>

#include <cuecc.h>

int main(int argc, char* argv[]) {
	int status;
	uint64_t total_errors[2];
	unsigned long long delay;

	cuInit(0);

	int driver_version;
	assert(cuDriverGetVersion(&driver_version) == CUDA_SUCCESS);

	printf("CUDA driver version: %d\n", driver_version);

	CUdevice dev;
	assert(cuDeviceGet(&dev, 0) == CUDA_SUCCESS);

	CUcontext ctx;
	assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);

	//set current context
	assert(cuCtxSetCurrent(ctx) == CUDA_SUCCESS);

	uint64_t* A = malloc(sizeof(uint64_t) * 31);
	if (A == NULL) {
		perror("malloc()");
		return 1;
	}
	for (uint64_t i = 0; i < 31; i++) {
		A[i] = i;
	}

	uint32_t* B = malloc(sizeof(uint32_t) * 31);
	if (B == NULL) {
		perror("malloc()");
		return 1;
	}

	uint16_t* C = malloc(sizeof(uint16_t) * 31);
	if (C == NULL) {
		perror("malloc()");
		return 1;
	}

	CUdeviceptr d_A;
	assert(cuMemAlloc(&d_A, sizeof(uint64_t) * 31) == CUDA_SUCCESS);
	assert(cuMemcpyHtoD(d_A, A, sizeof(uint64_t) * 31) == CUDA_SUCCESS);

	//copy uninitialized values to device
	CUdeviceptr d_B;
	assert(cuMemAlloc(&d_B, sizeof(uint32_t) * 31) == CUDA_SUCCESS);
	assert(cuMemcpyHtoD(d_B, B, sizeof(uint32_t) * 31) == CUDA_SUCCESS);

	//copy uninitialized values to device
	CUdeviceptr d_C;
	assert(cuMemAlloc(&d_C, sizeof(uint16_t) * 31) == CUDA_SUCCESS);
	assert(cuMemcpyHtoD(d_C, C, sizeof(uint16_t) * 31) == CUDA_SUCCESS);

	cudaECCMemoryObject_t mem_A;
	CUdeviceptr parity_A;
	CUdeviceptr parity_B;

	cudaECCHandle_t handle;

	assert(cuECCInit(&handle) == CUDA_EDAC_SUCCESS);

	assert(cuECCSetPreferredHsiao_77_22_Version(handle, 2) == 0);

	//default memory scrubbing interval is 300 seconds
	assert(cuECCGetMemoryScrubbingInterval(handle, &delay) == 0);
	assert(delay == 300);
	assert(cuECCSetMemoryScrubbingInterval(handle, 5) == 0);
	assert(cuECCGetMemoryScrubbingInterval(handle, &delay) == 0);
	assert(delay == 5);

	assert(cuECCAddMemoryObject(handle, d_A, &mem_A) == CUDA_EDAC_SUCCESS);
	assert(cuECCAddMemoryObject(handle, d_B, NULL) == CUDA_EDAC_SUCCESS);
	assert(cuECCAddMemoryObject(handle, d_C, NULL) == CUDA_EDAC_SUCCESS);

	//lock EDAC mutex
	assert(cuECCLockEDACMutex(handle) == 0);

	assert(cuECCUpdateMemObject(handle, mem_A) == CUDA_EDAC_SUCCESS);

	assert(cuECCGetMemObjectParityBits(handle, mem_A, &parity_A) == CUDA_EDAC_SUCCESS);

	assert(cuECCUpdateMemObjectWithDevicePointer(handle, d_B) == CUDA_EDAC_SUCCESS);

	assert(cuECCGetMemObjectParityBitsWithDevicePointer(handle, d_B, &parity_B) == CUDA_EDAC_SUCCESS);

	//unlock EDAC mutex
	assert(cuECCUnlockEDACMutex(handle) == 0);

	assert(cuECCGetTotalErrorsWithDevicePointer(handle, d_B, total_errors, 2 * sizeof(uint64_t)) == 0);

	assert(cuECCGetTotalErrors(handle, mem_A, total_errors, 2 * sizeof(uint64_t)) == 0);

	assert(cuECCRemoveMemObjectWithDevicePointer(handle, d_C) == CUDA_EDAC_SUCCESS);
	assert(cuECCRemoveMemObject(handle, mem_A) == CUDA_EDAC_SUCCESS);

	cuECCDestroy(handle);

	assert(cuMemFree(d_C) == CUDA_SUCCESS);
	assert(cuMemFree(d_B) == CUDA_SUCCESS);
	assert(cuMemFree(d_A) == CUDA_SUCCESS);
	assert(cuDevicePrimaryCtxRelease(dev) == CUDA_SUCCESS);

	free(C);
	free(B);
	free(A);

	return 0;
}
