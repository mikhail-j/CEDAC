/* Encode and decode using the Hsiao(39, 32) code in CUDA
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

extern "C" {

__device__ void fix_single_bit_error_hsiao_39_32(unsigned int* data, unsigned char* parity) {
	unsigned int new_data;
#pragma unroll
	for (int i = 0; i < 32; i++) {
		new_data = (*data) ^ (1U << i);
		if (!(((__popc(0xc14840ff & new_data) & 1) ^ (__popc(0x01 & (*parity)) & 1))
			| ((__popc(0x2124ff90 & new_data) & 1) ^ (__popc(0x02 & (*parity)) & 1))
			| ((__popc(0x6cff0808 & new_data) & 1) ^ (__popc(0x04 & (*parity)) & 1))
			| ((__popc(0xff01a444 & new_data) & 1) ^ (__popc(0x08 & (*parity)) & 1))
			| ((__popc(0x16f092a6 & new_data) & 1) ^ (__popc(0x10 & (*parity)) & 1))
			| ((__popc(0x101f7161 & new_data) & 1) ^ (__popc(0x20 & (*parity)) & 1))
			| ((__popc(0x8a820f1b & new_data) & 1) ^ (__popc(0x40 & (*parity)) & 1)))) {
			*data = new_data;
			return;
		}
	}
	unsigned char new_parity;
#pragma unroll
	for (int i = 0; i < 7; i++) {
		new_parity = (*parity) ^ (1 << i);
		if (!(((__popc(0xc14840ff & (*data)) & 1) ^ (__popc(0x01 & new_parity) & 1))
			| ((__popc(0x2124ff90 & (*data)) & 1) ^ (__popc(0x02 & new_parity) & 1))
			| ((__popc(0x6cff0808 & (*data)) & 1) ^ (__popc(0x04 & new_parity) & 1))
			| ((__popc(0xff01a444 & (*data)) & 1) ^ (__popc(0x08 & new_parity) & 1))
			| ((__popc(0x16f092a6 & (*data)) & 1) ^ (__popc(0x10 & new_parity) & 1))
			| ((__popc(0x101f7161 & (*data)) & 1) ^ (__popc(0x20 & new_parity) & 1))
			| ((__popc(0x8a820f1b & (*data)) & 1) ^ (__popc(0x40 & new_parity) & 1)))) {
			*parity = new_parity;
			return;
		}
	}
	return;
}

__device__ void syndrome_decoding_hsiao_39_32(unsigned int* data, unsigned char* parity, unsigned long long* errors) {
	if (((__popc(0xc14840ff & (*data)) & 1) ^ (__popc(0x01 & (*parity)) & 1))
			| ((__popc(0x2124ff90 & (*data)) & 1) ^ (__popc(0x02 & (*parity)) & 1))
			| ((__popc(0x6cff0808 & (*data)) & 1) ^ (__popc(0x04 & (*parity)) & 1))
			| ((__popc(0xff01a444 & (*data)) & 1) ^ (__popc(0x08 & (*parity)) & 1))
			| ((__popc(0x16f092a6 & (*data)) & 1) ^ (__popc(0x10 & (*parity)) & 1))
			| ((__popc(0x101f7161 & (*data)) & 1) ^ (__popc(0x20 & (*parity)) & 1))
			| ((__popc(0x8a820f1b & (*data)) & 1) ^ (__popc(0x40 & (*parity)) & 1))) {
		if (((__popc(0xc14840ff & (*data)) & 1) ^ (__popc(0x01 & (*parity)) & 1))
			^ ((__popc(0x2124ff90 & (*data)) & 1) ^ (__popc(0x02 & (*parity)) & 1))
			^ ((__popc(0x6cff0808 & (*data)) & 1) ^ (__popc(0x04 & (*parity)) & 1))
			^ ((__popc(0xff01a444 & (*data)) & 1) ^ (__popc(0x08 & (*parity)) & 1))
			^ ((__popc(0x16f092a6 & (*data)) & 1) ^ (__popc(0x10 & (*parity)) & 1))
			^ ((__popc(0x101f7161 & (*data)) & 1) ^ (__popc(0x20 & (*parity)) & 1))
			^ ((__popc(0x8a820f1b & (*data)) & 1) ^ (__popc(0x40 & (*parity)) & 1))) {
			fix_single_bit_error_hsiao_39_32(data, parity);
			atomicAdd(errors, 1ULL);
			return;
		}
		else {
			atomicAdd(errors + 1, 1ULL);
			return;
		}
	}
	return;
}

__global__ void generate_parity_hsiao_39_32(unsigned int* data, unsigned char* parity, size_t data_size) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	while (i < data_size) {
		parity[i] = ((__popc(0xc14840ff & data[i]) & 1)
			+ ((__popc(0x2124ff90 & data[i]) & 1) << 1)
			+ ((__popc(0x6cff0808 & data[i]) & 1) << 2)
			+ ((__popc(0xff01a444 & data[i]) & 1) << 3)
			+ ((__popc(0x16f092a6 & data[i]) & 1) << 4)
			+ ((__popc(0x101f7161 & data[i]) & 1) << 5)
			+ ((__popc(0x8a820f1b & data[i]) & 1) << 6));
		i = i + (blockDim.x * gridDim.x);
	}

	return;
}

__global__ void edac_hsiao_39_32(unsigned int* data, unsigned char* parity, size_t data_size, unsigned long long* error_count) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	while (i < data_size) {
		syndrome_decoding_hsiao_39_32(data + i, parity + i, error_count);
		i = i + (blockDim.x * gridDim.x);
	}

	return;
}

}
