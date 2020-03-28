/* Encode and decode using the Hsiao(72, 64) version 2 code in CUDA
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

__device__ void fix_single_bit_error_hsiao_72_64_v2(unsigned long long* data, unsigned char* parity) {
	unsigned long long new_data;
#pragma unroll
	for (int i = 0; i < 64; i++) {
		new_data = (*data) ^ (1ULL << i);
		if (!(((__popcll(0x0111111630f0f0ff & new_data) & 1) ^ (__popc(0x01 & (*parity)) & 1))
			| ((__popcll(0x02222226cf00ff0f & new_data) & 1) ^ (__popc(0x02 & (*parity)) & 1))
			| ((__popcll(0x64444440f0ff0f0c & new_data) & 1) ^ (__popc(0x04 & (*parity)) & 1))
			| ((__popcll(0x68888880ff0f00f3 & new_data) & 1) ^ (__popc(0x08 & (*parity)) & 1))
			| ((__popcll(0xcf00f0ff01111116 & new_data) & 1) ^ (__popc(0x10 & (*parity)) & 1))
			| ((__popcll(0x30f0ff0f02222226 & new_data) & 1) ^ (__popc(0x20 & (*parity)) & 1))
			| ((__popcll(0xf0ff00f364444440 & new_data) & 1) ^ (__popc(0x40 & (*parity)) & 1))
			| ((__popcll(0xff0f0f0c68888880 & new_data) & 1) ^ (__popc(0x80 & (*parity)) & 1)))) {
			*data = new_data;
			return;
		}
	}
	unsigned char new_parity;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		new_parity = (*parity) ^ (1 << i);
		if (!(((__popcll(0x0111111630f0f0ff & (*data)) & 1) ^ (__popc(0x01 & new_parity) & 1))
			| ((__popcll(0x02222226cf00ff0f & (*data)) & 1) ^ (__popc(0x02 & new_parity) & 1))
			| ((__popcll(0x64444440f0ff0f0c & (*data)) & 1) ^ (__popc(0x04 & new_parity) & 1))
			| ((__popcll(0x68888880ff0f00f3 & (*data)) & 1) ^ (__popc(0x08 & new_parity) & 1))
			| ((__popcll(0xcf00f0ff01111116 & (*data)) & 1) ^ (__popc(0x10 & new_parity) & 1))
			| ((__popcll(0x30f0ff0f02222226 & (*data)) & 1) ^ (__popc(0x20 & new_parity) & 1))
			| ((__popcll(0xf0ff00f364444440 & (*data)) & 1) ^ (__popc(0x40 & new_parity) & 1))
			| ((__popcll(0xff0f0f0c68888880 & (*data)) & 1) ^ (__popc(0x80 & new_parity) & 1)))) {
			*parity = new_parity;
			return;
		}
	}
	return;
}

__device__ void syndrome_decoding_hsiao_72_64_v2(unsigned long long* data, unsigned char* parity, unsigned long long* errors) {
	if (((__popcll(0x0111111630f0f0ff & (*data)) & 1) ^ (__popc(0x01 & (*parity)) & 1))
			| ((__popcll(0x02222226cf00ff0f & (*data)) & 1) ^ (__popc(0x02 & (*parity)) & 1))
			| ((__popcll(0x64444440f0ff0f0c & (*data)) & 1) ^ (__popc(0x04 & (*parity)) & 1))
			| ((__popcll(0x68888880ff0f00f3 & (*data)) & 1) ^ (__popc(0x08 & (*parity)) & 1))
			| ((__popcll(0xcf00f0ff01111116 & (*data)) & 1) ^ (__popc(0x10 & (*parity)) & 1))
			| ((__popcll(0x30f0ff0f02222226 & (*data)) & 1) ^ (__popc(0x20 & (*parity)) & 1))
			| ((__popcll(0xf0ff00f364444440 & (*data)) & 1) ^ (__popc(0x40 & (*parity)) & 1))
			| ((__popcll(0xff0f0f0c68888880 & (*data)) & 1) ^ (__popc(0x80 & (*parity)) & 1))) {
		if (((__popcll(0x0111111630f0f0ff & (*data)) & 1) ^ (__popc(0x01 & (*parity)) & 1))
			^ ((__popcll(0x02222226cf00ff0f & (*data)) & 1) ^ (__popc(0x02 & (*parity)) & 1))
			^ ((__popcll(0x64444440f0ff0f0c & (*data)) & 1) ^ (__popc(0x04 & (*parity)) & 1))
			^ ((__popcll(0x68888880ff0f00f3 & (*data)) & 1) ^ (__popc(0x08 & (*parity)) & 1))
			^ ((__popcll(0xcf00f0ff01111116 & (*data)) & 1) ^ (__popc(0x10 & (*parity)) & 1))
			^ ((__popcll(0x30f0ff0f02222226 & (*data)) & 1) ^ (__popc(0x20 & (*parity)) & 1))
			^ ((__popcll(0xf0ff00f364444440 & (*data)) & 1) ^ (__popc(0x40 & (*parity)) & 1))
			^ ((__popcll(0xff0f0f0c68888880 & (*data)) & 1) ^ (__popc(0x80 & (*parity)) & 1))) {
			fix_single_bit_error_hsiao_72_64_v2(data, parity);
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

__global__ void generate_parity_hsiao_72_64_v2(unsigned long long* data, unsigned char* parity, size_t data_size) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	while (i < data_size) {
		parity[i] = ((__popcll(0x0111111630f0f0ff & data[i]) & 1)
			+ ((__popcll(0x02222226cf00ff0f & data[i]) & 1) << 1)
			+ ((__popcll(0x64444440f0ff0f0c & data[i]) & 1) << 2)
			+ ((__popcll(0x68888880ff0f00f3 & data[i]) & 1) << 3)
			+ ((__popcll(0xcf00f0ff01111116 & data[i]) & 1) << 4)
			+ ((__popcll(0x30f0ff0f02222226 & data[i]) & 1) << 5)
			+ ((__popcll(0xf0ff00f364444440 & data[i]) & 1) << 6)
			+ ((__popcll(0xff0f0f0c68888880 & data[i]) & 1) << 7));
		i = i + (blockDim.x * gridDim.x);
	}

	return;
}

__global__ void edac_hsiao_72_64_v2(unsigned long long* data, unsigned char* parity, size_t data_size, unsigned long long* error_count) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	while (i < data_size) {
		syndrome_decoding_hsiao_72_64_v2(data + i, parity + i, error_count);
		i = i + (blockDim.x * gridDim.x);
	}

	return;
}

}
