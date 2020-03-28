/* Encode and decode using the Hsiao(22, 16) code in OpenCL
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

void fix_single_bit_error_hsiao_22_16(__global ushort* data, __global uchar* parity) {
	__local ushort new_data;
#pragma unroll
	for (int i = 0; i < 16; i++) {
		new_data = (*data) ^ (1 << i);
		if (!(((popcount(0x443f & new_data) & 1) ^ (popcount(0x01 & (*parity)) & 1))
			| ((popcount(0x13c7 & new_data) & 1) ^ (popcount(0x02 & (*parity)) & 1))
			| ((popcount(0xe1d1 & new_data) & 1) ^ (popcount(0x04 & (*parity)) & 1))
			| ((popcount(0xee60 & new_data) & 1) ^ (popcount(0x08 & (*parity)) & 1))
			| ((popcount(0x3e8a & new_data) & 1) ^ (popcount(0x10 & (*parity)) & 1))
			| ((popcount(0x993c & new_data) & 1) ^ (popcount(0x20 & (*parity)) & 1)))) {
			*data = new_data;
			return;
		}
	}
	__local uchar new_parity;
#pragma unroll
	for (int j = 0; j < 6; j++) {
		new_parity = (*parity) ^ (1 << j);
		if (!(((popcount(0x443f & (*data)) & 1) ^ (popcount(0x01 & new_parity) & 1))
			| ((popcount(0x13c7 & (*data)) & 1) ^ (popcount(0x02 & new_parity) & 1))
			| ((popcount(0xe1d1 & (*data)) & 1) ^ (popcount(0x04 & new_parity) & 1))
			| ((popcount(0xee60 & (*data)) & 1) ^ (popcount(0x08 & new_parity) & 1))
			| ((popcount(0x3e8a & (*data)) & 1) ^ (popcount(0x10 & new_parity) & 1))
			| ((popcount(0x993c & (*data)) & 1) ^ (popcount(0x20 & new_parity) & 1)))) {
			*parity = new_parity;
			return;
		}
	}
	return;
}

void syndrome_decoding_hsiao_22_16(__global ushort* data, __global uchar* parity, __global ulong* errors) {
	if (((popcount(0x443f & (*data)) & 1) ^ (popcount(0x01 & (*parity)) & 1))
			| ((popcount(0x13c7 & (*data)) & 1) ^ (popcount(0x02 & (*parity)) & 1))
			| ((popcount(0xe1d1 & (*data)) & 1) ^ (popcount(0x04 & (*parity)) & 1))
			| ((popcount(0xee60 & (*data)) & 1) ^ (popcount(0x08 & (*parity)) & 1))
			| ((popcount(0x3e8a & (*data)) & 1) ^ (popcount(0x10 & (*parity)) & 1))
			| ((popcount(0x993c & (*data)) & 1) ^ (popcount(0x20 & (*parity)) & 1))) {
		if ((((popcount(0x443f & (*data)) & 1) ^ (popcount(0x01 & (*parity)) & 1))
			^ ((popcount(0x13c7 & (*data)) & 1) ^ (popcount(0x02 & (*parity)) & 1))
			^ ((popcount(0xe1d1 & (*data)) & 1) ^ (popcount(0x04 & (*parity)) & 1))
			^ ((popcount(0xee60 & (*data)) & 1) ^ (popcount(0x08 & (*parity)) & 1))
			^ ((popcount(0x3e8a & (*data)) & 1) ^ (popcount(0x10 & (*parity)) & 1))
			^ ((popcount(0x993c & (*data)) & 1) ^ (popcount(0x20 & (*parity)) & 1)))) {
			fix_single_bit_error_hsiao_22_16(data, parity);
			atom_inc(errors);
		}
		else {
			atom_inc(errors + 1);
		}
	}
	return;
}

__kernel void generate_parity_hsiao_22_16(__global ushort * data, __global uchar* parity, ulong data_size) {
	__local ulong i;
	i = (get_local_size(0) * get_group_id(0)) + get_local_id(0);
	while (i < data_size) {
		parity[i] = ((popcount(0x443f & data[i]) & 1)
				+ ((popcount(0x13c7 & data[i]) & 1) << 1)
				+ ((popcount(0xe1d1 & data[i]) & 1) << 2)
				+ ((popcount(0xee60 & data[i]) & 1) << 3)
				+ ((popcount(0x3e8a & data[i]) & 1) << 4)
				+ ((popcount(0x993c & data[i]) & 1) << 5));
		i = i + (get_local_size(0) * get_num_groups(0));
	}
	return;
}

__kernel void edac_hsiao_22_16(__global ushort* data, __global uchar* parity, ulong data_size, __global ulong* errors) {
	__local ulong i;
	i = (get_local_size(0) * get_group_id(0)) + get_local_id(0);
	while (i < data_size) {
		syndrome_decoding_hsiao_22_16(data + i, parity + i, errors);
		i = i + (get_local_size(0) * get_num_groups(0));
	}
	return;
}
