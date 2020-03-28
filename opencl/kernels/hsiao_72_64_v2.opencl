/* Encode and decode using the Hsiao(72, 64) version 2 code in OpenCL
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
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

void fix_single_bit_error_hsiao_72_64_v2(__global ulong* data, __global uchar* parity) {
	__local ulong new_data;
#pragma unroll
	for (int i = 0; i < 64; i++) {
		new_data = (*data) ^ (1UL << i);
		if (!(((popcount(0x0111111630f0f0ff & new_data) & 1) ^ (popcount(0x01 & (*parity)) & 1))
			| ((popcount(0x02222226cf00ff0f & new_data) & 1) ^ (popcount(0x02 & (*parity)) & 1))
			| ((popcount(0x64444440f0ff0f0c & new_data) & 1) ^ (popcount(0x04 & (*parity)) & 1))
			| ((popcount(0x68888880ff0f00f3 & new_data) & 1) ^ (popcount(0x08 & (*parity)) & 1))
			| ((popcount(0xcf00f0ff01111116 & new_data) & 1) ^ (popcount(0x10 & (*parity)) & 1))
			| ((popcount(0x30f0ff0f02222226 & new_data) & 1) ^ (popcount(0x20 & (*parity)) & 1))
			| ((popcount(0xf0ff00f364444440 & new_data) & 1) ^ (popcount(0x40 & (*parity)) & 1))
			| ((popcount(0xff0f0f0c68888880 & new_data) & 1) ^ (popcount(0x80 & (*parity)) & 1)))) {
			*data = new_data;
			return;
		}
	}
	__local uchar new_parity;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		new_parity = *parity ^ (1 << i);
		if (!(((popcount(0x0111111630f0f0ff & (*data)) & 1) ^ (popcount(0x01 & new_parity) & 1))
			| ((popcount(0x02222226cf00ff0f & (*data)) & 1) ^ (popcount(0x02 & new_parity) & 1))
			| ((popcount(0x64444440f0ff0f0c & (*data)) & 1) ^ (popcount(0x04 & new_parity) & 1))
			| ((popcount(0x68888880ff0f00f3 & (*data)) & 1) ^ (popcount(0x08 & new_parity) & 1))
			| ((popcount(0xcf00f0ff01111116 & (*data)) & 1) ^ (popcount(0x10 & new_parity) & 1))
			| ((popcount(0x30f0ff0f02222226 & (*data)) & 1) ^ (popcount(0x20 & new_parity) & 1))
			| ((popcount(0xf0ff00f364444440 & (*data)) & 1) ^ (popcount(0x40 & new_parity) & 1))
			| ((popcount(0xff0f0f0c68888880 & (*data)) & 1) ^ (popcount(0x80 & new_parity) & 1)))) {
			*parity = new_parity;
			return;
		}
	}
	return;
}

void syndrome_decoding_hsiao_72_64_v2(__global ulong* data, __global uchar* parity, __global ulong* errors) {
	if (((popcount(0x0111111630f0f0ff & (*data)) & 1) ^ (popcount(0x01 & (*parity)) & 1))
			| ((popcount(0x02222226cf00ff0f & (*data)) & 1) ^ (popcount(0x02 & (*parity)) & 1))
			| ((popcount(0x64444440f0ff0f0c & (*data)) & 1) ^ (popcount(0x04 & (*parity)) & 1))
			| ((popcount(0x68888880ff0f00f3 & (*data)) & 1) ^ (popcount(0x08 & (*parity)) & 1))
			| ((popcount(0xcf00f0ff01111116 & (*data)) & 1) ^ (popcount(0x10 & (*parity)) & 1))
			| ((popcount(0x30f0ff0f02222226 & (*data)) & 1) ^ (popcount(0x20 & (*parity)) & 1))
			| ((popcount(0xf0ff00f364444440 & (*data)) & 1) ^ (popcount(0x40 & (*parity)) & 1))
			| ((popcount(0xff0f0f0c68888880 & (*data)) & 1) ^ (popcount(0x80 & (*parity)) & 1))) {
		if (((popcount(0x0111111630f0f0ff & (*data)) & 1) ^ (popcount(0x01 & (*parity)) & 1))
			^ ((popcount(0x02222226cf00ff0f & (*data)) & 1) ^ (popcount(0x02 & (*parity)) & 1))
			^ ((popcount(0x64444440f0ff0f0c & (*data)) & 1) ^ (popcount(0x04 & (*parity)) & 1))
			^ ((popcount(0x68888880ff0f00f3 & (*data)) & 1) ^ (popcount(0x08 & (*parity)) & 1))
			^ ((popcount(0xcf00f0ff01111116 & (*data)) & 1) ^ (popcount(0x10 & (*parity)) & 1))
			^ ((popcount(0x30f0ff0f02222226 & (*data)) & 1) ^ (popcount(0x20 & (*parity)) & 1))
			^ ((popcount(0xf0ff00f364444440 & (*data)) & 1) ^ (popcount(0x40 & (*parity)) & 1))
			^ ((popcount(0xff0f0f0c68888880 & (*data)) & 1) ^ (popcount(0x80 & (*parity)) & 1))) {
			fix_single_bit_error_hsiao_72_64_v2(data, parity);
			atom_inc(errors);
		}
		else {
			atom_inc(errors + 1);
		}
	}
	return;
}

__kernel void generate_parity_hsiao_72_64_v2(__global ulong * data, __global uchar* parity, ulong data_size) {
	__local ulong i;
	i = (get_local_size(0) * get_group_id(0)) + get_local_id(0);
	while (i < data_size) {
		parity[i] = ((popcount(0x0111111630f0f0ff & data[i]) & 1)
				+ ((popcount(0x02222226cf00ff0f & data[i]) & 1) << 1)
				+ ((popcount(0x64444440f0ff0f0c & data[i]) & 1) << 2)
				+ ((popcount(0x68888880ff0f00f3 & data[i]) & 1) << 3)
				+ ((popcount(0xcf00f0ff01111116 & data[i]) & 1) << 4)
				+ ((popcount(0x30f0ff0f02222226 & data[i]) & 1) << 5)
				+ ((popcount(0xf0ff00f364444440 & data[i]) & 1) << 6)
				+ ((popcount(0xff0f0f0c68888880 & data[i]) & 1) << 7));
		i = i + (get_local_size(0) * get_num_groups(0));
	}
	return;
}

__kernel void edac_hsiao_72_64_v2(__global ulong* data, __global uchar* parity, ulong data_size, __global ulong* errors) {
	__local ulong i;
	i = (get_local_size(0) * get_group_id(0)) + get_local_id(0);
	while (i < data_size) {
		syndrome_decoding_hsiao_72_64_v2(data + i, parity + i, errors);
		i = i + (get_local_size(0) * get_num_groups(0));
	}
	return;
}
