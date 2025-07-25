/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2015 Cavium, Inc
 */

#include "acl_run_rvv.h"

int
rte_acl_classify_rvv(const struct rte_acl_ctx *ctx, const uint8_t **data,
		      uint32_t *results, uint32_t num, uint32_t categories)
{
	if (likely(num >= 8))
		return search_rvv_8(ctx, data, results, num, categories);
	else if (num >= 4)
		return search_rvv_4(ctx, data, results, num, categories);
	else
		return rte_acl_classify_scalar(ctx, data, results, num,
			categories);
}
