/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2022 StarFive
 * Copyright(c) 2022 SiFive
 * Copyright(c) 2022 Semihalf
 */

#ifndef RTE_VECT_RISCV_H
#define RTE_VECT_RISCV_H

#include <stdint.h>
#include "generic/rte_vect.h"
#include "rte_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RTE_VECT_DEFAULT_SIMD_BITWIDTH RTE_VECT_SIMD_256

typedef int32_t		xmm_t __attribute__((vector_size(16)));

#define XMM_SIZE	(sizeof(xmm_t))
#define XMM_MASK	(XMM_SIZE - 1)

typedef union __rte_aligned(16) rte_xmm {
	xmm_t		x;
	uint8_t		u8[XMM_SIZE / sizeof(uint8_t)];
	uint16_t	u16[XMM_SIZE / sizeof(uint16_t)];
	uint32_t	u32[XMM_SIZE / sizeof(uint32_t)];
	uint64_t	u64[XMM_SIZE / sizeof(uint64_t)];
	double		pd[XMM_SIZE / sizeof(double)];
} rte_xmm_t;

static inline xmm_t
vect_load_128(void *p)
{
	xmm_t ret = *((xmm_t *)p);
	return ret;
}

static inline xmm_t
vect_and(xmm_t data, xmm_t mask)
{
	rte_xmm_t ret = {.x = data };
	rte_xmm_t m = {.x = mask };
	ret.u64[0] &= m.u64[0];
	ret.u64[1] &= m.u64[1];
	return ret.x;
}

#ifdef __cplusplus
}
#endif

#endif /* RTE_VECT_RISCV_H */
