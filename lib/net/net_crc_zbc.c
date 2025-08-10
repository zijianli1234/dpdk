/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) ByteDance 2024
 */

#include <riscv_bitmanip.h>
#include <stdint.h>

#include <rte_common.h>
#include <rte_net_crc.h>
#include <stdio.h>

#include "net_crc.h"

/* CLMUL CRC computation context structure */
struct crc_clmul_ctx {
	uint64_t Pr;
	uint64_t mu;
	uint64_t k3;
	uint64_t k4;
	uint64_t k5;
};

struct crc_clmul_ctx crc32_eth_clmul;
struct crc_clmul_ctx crc16_ccitt_clmul;

/* Perform Barrett's reduction on 8, 16, 32 or 64-bit value */
static inline uint32_t
crc32_barrett_zbc(
	const uint64_t data,
	uint32_t crc,
	uint32_t bits,
	const struct crc_clmul_ctx *params)
{
	assert((bits == 64) || (bits == 32) || (bits == 16) || (bits == 8));

	/* Combine data with the initial value */
	uint64_t temp = (uint64_t)(data ^ crc) << (64 - bits);

	/*
	 * Multiply by mu, which is 2^96 / P. Division by 2^96 occurs by taking
	 * the lower 64 bits of the result (remember we're inverted)
	 */
	temp = __riscv_clmul_64(temp, params->mu);
	/* Multiply by P */
	temp = __riscv_clmulh_64(temp, params->Pr);

	/* Subtract from original (only needed for smaller sizes) */
	if (bits == 16 || bits == 8)
		temp ^= crc >> bits;

	return temp;
}

/* Repeat Barrett's reduction for short buffer sizes */
static inline uint32_t
crc32_repeated_barrett_zbc(
	const uint8_t *data,
	uint32_t data_len,
	uint32_t crc,
	const struct crc_clmul_ctx *params)
{
	while (data_len >= 8) {
		crc = crc32_barrett_zbc(*(const uint64_t *)data, crc, 64, params);
		data += 8;
		data_len -= 8;
	}
	if (data_len >= 4) {
		crc = crc32_barrett_zbc(*(const uint32_t *)data, crc, 32, params);
		data += 4;
		data_len -= 4;
	}
	if (data_len >= 2) {
		crc = crc32_barrett_zbc(*(const uint16_t *)data, crc, 16, params);
		data += 2;
		data_len -= 2;
	}
	if (data_len >= 1)
		crc = crc32_barrett_zbc(*(const uint8_t *)data, crc, 8, params);

	return crc;
}

/* Perform a reduction by 1 on a buffer (minimum length 2) */
static inline void
crc32_reduce_zbc(const uint64_t *data, uint64_t *high, uint64_t *low,
		 const struct crc_clmul_ctx *params)
{
	uint64_t highh = __riscv_clmulh_64(params->k3, *high);
	uint64_t highl = __riscv_clmul_64(params->k3, *high);
	uint64_t lowh = __riscv_clmulh_64(params->k4, *low);
	uint64_t lowl = __riscv_clmul_64(params->k4, *low);

	*high = highl ^ lowl;
	*low = highh ^ lowh;

	*high ^= *(data++);
	*low ^= *(data++);
}

static inline uint32_t
crc32_eth_calc_zbc(
	const uint8_t *data,
	uint32_t data_len,
	uint32_t crc,
	const struct crc_clmul_ctx *params)
{
	uint64_t high, low;
	/* Minimum length we can do reduction-by-1 over */
	const uint32_t min_len = 16;
	/* Barrett reduce until buffer aligned to 8-byte word */
	uint32_t misalign = (size_t)data & 7;
	if (misalign != 0 && misalign <= data_len) {
		crc = crc32_repeated_barrett_zbc(data, misalign, crc, params);
		data += misalign;
		data_len -= misalign;
	}

	if (data_len < min_len)
		return crc32_repeated_barrett_zbc(data, data_len, crc, params);

	/* Fold buffer into two 8-byte words */
	high = *((const uint64_t *)data) ^ crc;
	low = *((const uint64_t *)(data + 8));
	data += 16;
	data_len -= 16;

	for (; data_len >= 16; data_len -= 16, data += 16)
		crc32_reduce_zbc((const uint64_t *)data, &high, &low, params);

	/* Fold last 128 bits into 96 */
	low = __riscv_clmul_64(params->k4, high) ^ low;
	high = __riscv_clmulh_64(params->k4, high);
	/* Upper 32 bits of high are now zero */
	high = (low >> 32) | (high << 32);

	/* Fold last 96 bits into 64 */
	low = __riscv_clmul_64(low & 0xffffffff, params->k5);
	low ^= high;

	/*
	 * Barrett reduction of remaining 64 bits, using high to store initial
	 * value of low
	 */
	high = low;
	low = __riscv_clmul_64(low, params->mu);
	low &= 0xffffffff;
	low = __riscv_clmul_64(low, params->Pr);
	crc = (high ^ low) >> 32;

	/* Combine crc with any excess */
	crc = crc32_repeated_barrett_zbc(data, data_len, crc, params);

	return crc;
}

void
rte_net_crc_zbc_init(void)
{
	/* Initialise CRC32 data */
	crc32_eth_clmul.Pr = 0x1db710641LL; /* polynomial P reversed */
	crc32_eth_clmul.mu = 0xb4e5b025f7011641LL; /* (2 ^ 64 / P) reversed */
	crc32_eth_clmul.k3 = 0x1751997d0LL; /* (x^(128+32) mod P << 32) reversed << 1 */
	crc32_eth_clmul.k4 = 0x0ccaa009eLL; /* (x^(128-32) mod P << 32) reversed << 1 */
	crc32_eth_clmul.k5 = 0x163cd6124LL; /* (x^64 mod P << 32) reversed << 1 */

	/* Initialise CRC16 data */
	/* Same calculations as above, with polynomial << 16 */
	crc16_ccitt_clmul.Pr = 0x10811LL;
	crc16_ccitt_clmul.mu = 0x859b040b1c581911LL;
	crc16_ccitt_clmul.k3 = 0x8e10LL;
	crc16_ccitt_clmul.k4 = 0x189aeLL;
	crc16_ccitt_clmul.k5 = 0x114aaLL;
}

uint32_t
rte_crc16_ccitt_zbc_handler(const uint8_t *data, uint32_t data_len)
{
	/* Negate the crc, which is present in the lower 16-bits */
	return (uint16_t)~crc32_eth_calc_zbc(data,
		data_len,
		0xffff,
		&crc16_ccitt_clmul);
}

uint32_t
rte_crc32_eth_zbc_handler(const uint8_t *data, uint32_t data_len)
{
	return ~crc32_eth_calc_zbc(data,
		data_len,
		0xffffffffUL,
		&crc32_eth_clmul);
}