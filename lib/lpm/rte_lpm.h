/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2014 Intel Corporation
 * Copyright(c) 2020 Arm Limited
 */

#ifndef _RTE_LPM_H_
#define _RTE_LPM_H_

/**
 * @file
 * RTE Longest Prefix Match (LPM)
 */

#include <errno.h>
#include <stdalign.h>
#include <stdint.h>

#include <rte_branch_prediction.h>
#include <rte_byteorder.h>
#include <rte_common.h>
#include <rte_vect.h>
#include <rte_rcu_qsbr.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Max number of characters in LPM name. */
#define RTE_LPM_NAMESIZE                32

/** Maximum depth value possible for IPv4 LPM. */
#define RTE_LPM_MAX_DEPTH               32

/** @internal Total number of tbl24 entries. */
#define RTE_LPM_TBL24_NUM_ENTRIES       (1 << 24)

/** @internal Number of entries in a tbl8 group. */
#define RTE_LPM_TBL8_GROUP_NUM_ENTRIES  256

/** @internal Max number of tbl8 groups in the tbl8. */
#define RTE_LPM_MAX_TBL8_NUM_GROUPS         (1 << 24)

/** @internal Total number of tbl8 groups in the tbl8. */
#define RTE_LPM_TBL8_NUM_GROUPS         256

/** @internal Total number of tbl8 entries. */
#define RTE_LPM_TBL8_NUM_ENTRIES        (RTE_LPM_TBL8_NUM_GROUPS * \
					RTE_LPM_TBL8_GROUP_NUM_ENTRIES)

/** @internal Macro to enable/disable run-time checks. */
#if defined(RTE_LIBRTE_LPM_DEBUG)
#define RTE_LPM_RETURN_IF_TRUE(cond, retval) do { \
	if (cond) return (retval);                \
} while (0)
#else
#define RTE_LPM_RETURN_IF_TRUE(cond, retval)
#endif

/** @internal bitmask with valid and valid_group fields set */
#define RTE_LPM_VALID_EXT_ENTRY_BITMASK 0x03000000

/** Bitmask used to indicate successful lookup */
#define RTE_LPM_LOOKUP_SUCCESS          0x01000000

/** @internal Default RCU defer queue entries to reclaim in one go. */
#define RTE_LPM_RCU_DQ_RECLAIM_MAX	16

/** RCU reclamation modes */
enum rte_lpm_qsbr_mode {
	/** Create defer queue for reclaim. */
	RTE_LPM_QSBR_MODE_DQ = 0,
	/** Use blocking mode reclaim. No defer queue created. */
	RTE_LPM_QSBR_MODE_SYNC
};

#if RTE_BYTE_ORDER == RTE_LITTLE_ENDIAN
/** @internal Tbl24 entry structure. */
__extension__
struct rte_lpm_tbl_entry {
	union {
		RTE_ATOMIC(uint32_t) val;
		struct {
			/**
			 * Stores Next hop (tbl8 or tbl24 when valid_group is not set) or
			 * a group index pointing to a tbl8 structure (tbl24 only, when
			 * valid_group is set)
			 */
			uint32_t next_hop    :24;
			/* Using single uint8_t to store 3 values. */
			uint32_t valid       :1;   /**< Validation flag. */
			/**
			 * For tbl24:
			 *  - valid_group == 0: entry stores a next hop
			 *  - valid_group == 1: entry stores a group_index pointing to a tbl8
			 * For tbl8:
			 *  - valid_group indicates whether the current tbl8 is in use or not
			 */
			uint32_t valid_group :1;
			uint32_t depth       :6; /**< Rule depth. */
		};
	};
};

#else

__extension__
struct rte_lpm_tbl_entry {
	union {
		RTE_ATOMIC(uint32_t) val;
		struct {
			uint32_t depth       :6;
			uint32_t valid_group :1;
			uint32_t valid       :1;
			uint32_t next_hop    :24;
		};
	};
};

#endif

static_assert(sizeof(struct rte_lpm_tbl_entry) == sizeof(uint32_t),
		"sizeof(struct rte_lpm_tbl_entry) == sizeof(uint32_t)");

/** LPM configuration structure. */
struct rte_lpm_config {
	uint32_t max_rules;      /**< Max number of rules. */
	uint32_t number_tbl8s;   /**< Number of tbl8s to allocate. */
	int flags;               /**< This field is currently unused. */
};

/** @internal LPM structure. */
struct rte_lpm {
	/* LPM Tables. */
	alignas(RTE_CACHE_LINE_SIZE) struct rte_lpm_tbl_entry tbl24[RTE_LPM_TBL24_NUM_ENTRIES];
			/**< LPM tbl24 table. */
	struct rte_lpm_tbl_entry *tbl8; /**< LPM tbl8 table. */
};

/** LPM RCU QSBR configuration structure. */
struct rte_lpm_rcu_config {
	struct rte_rcu_qsbr *v;	/* RCU QSBR variable. */
	/* Mode of RCU QSBR. RTE_LPM_QSBR_MODE_xxx
	 * '0' for default: create defer queue for reclaim.
	 */
	enum rte_lpm_qsbr_mode mode;
	uint32_t dq_size;	/* RCU defer queue size.
				 * default: lpm->number_tbl8s.
				 */
	uint32_t reclaim_thd;	/* Threshold to trigger auto reclaim. */
	uint32_t reclaim_max;	/* Max entries to reclaim in one go.
				 * default: RTE_LPM_RCU_DQ_RECLAIM_MAX.
				 */
};

/**
 * Free an LPM object.
 *
 * @param lpm
 *   LPM object handle
 *   If lpm is NULL, no operation is performed.
 */
void
rte_lpm_free(struct rte_lpm *lpm);

/**
 * Create an LPM object.
 *
 * @param name
 *   LPM object name
 * @param socket_id
 *   NUMA socket ID for LPM table memory allocation
 * @param config
 *   Structure containing the configuration
 * @return
 *   Handle to LPM object on success, NULL otherwise with rte_errno set
 *   to an appropriate values. Possible rte_errno values include:
 *    - E_RTE_NO_CONFIG - function could not get pointer to rte_config structure
 *    - E_RTE_SECONDARY - function was called from a secondary process instance
 *    - EINVAL - invalid parameter passed to function
 *    - ENOSPC - the maximum number of memzones has already been allocated
 *    - EEXIST - a memzone with the same name already exists
 *    - ENOMEM - no appropriate memory area found in which to create memzone
 */
struct rte_lpm *
rte_lpm_create(const char *name, int socket_id,
	       const struct rte_lpm_config *config)
	__rte_malloc __rte_dealloc(rte_lpm_free, 1);

/**
 * Find an existing LPM object and return a pointer to it.
 *
 * @param name
 *   Name of the lpm object as passed to rte_lpm_create()
 * @return
 *   Pointer to lpm object or NULL if object not found with rte_errno
 *   set appropriately. Possible rte_errno values include:
 *    - ENOENT - required entry not available to return.
 */
struct rte_lpm *
rte_lpm_find_existing(const char *name);

/**
 * Associate RCU QSBR variable with an LPM object.
 *
 * @param lpm
 *   the lpm object to add RCU QSBR
 * @param cfg
 *   RCU QSBR configuration
 * @return
 *   On success - 0
 *   On error - 1 with error code set in rte_errno.
 *   Possible rte_errno codes are:
 *   - EINVAL - invalid pointer
 *   - EEXIST - already added QSBR
 *   - ENOMEM - memory allocation failure
 */
int rte_lpm_rcu_qsbr_add(struct rte_lpm *lpm, struct rte_lpm_rcu_config *cfg);

/**
 * Add a rule to the LPM table.
 *
 * @param lpm
 *   LPM object handle
 * @param ip
 *   IP of the rule to be added to the LPM table
 * @param depth
 *   Depth of the rule to be added to the LPM table
 * @param next_hop
 *   Next hop of the rule to be added to the LPM table
 * @return
 *   0 on success, negative value otherwise
 */
int
rte_lpm_add(struct rte_lpm *lpm, uint32_t ip, uint8_t depth, uint32_t next_hop);

/**
 * Check if a rule is present in the LPM table,
 * and provide its next hop if it is.
 *
 * @param lpm
 *   LPM object handle
 * @param ip
 *   IP of the rule to be searched
 * @param depth
 *   Depth of the rule to searched
 * @param next_hop
 *   Next hop of the rule (valid only if it is found)
 * @return
 *   1 if the rule exists, 0 if it does not, a negative value on failure
 */
int
rte_lpm_is_rule_present(struct rte_lpm *lpm, uint32_t ip, uint8_t depth,
uint32_t *next_hop);

/**
 * Delete a rule from the LPM table.
 *
 * @param lpm
 *   LPM object handle
 * @param ip
 *   IP of the rule to be deleted from the LPM table
 * @param depth
 *   Depth of the rule to be deleted from the LPM table
 * @return
 *   0 on success, negative value otherwise
 */
int
rte_lpm_delete(struct rte_lpm *lpm, uint32_t ip, uint8_t depth);

/**
 * Delete all rules from the LPM table.
 *
 * @param lpm
 *   LPM object handle
 */
void
rte_lpm_delete_all(struct rte_lpm *lpm);

/**
 * Lookup an IP into the LPM table.
 *
 * @param lpm
 *   LPM object handle
 * @param ip
 *   IP to be looked up in the LPM table
 * @param next_hop
 *   Next hop of the most specific rule found for IP (valid on lookup hit only)
 * @return
 *   -EINVAL for incorrect arguments, -ENOENT on lookup miss, 0 on lookup hit
 */
static inline int
rte_lpm_lookup(const struct rte_lpm *lpm, uint32_t ip, uint32_t *next_hop)
{
	unsigned tbl24_index = (ip >> 8);
	uint32_t tbl_entry;
	const uint32_t *ptbl;

	/* DEBUG: Check user input arguments. */
	RTE_LPM_RETURN_IF_TRUE(((lpm == NULL) || (next_hop == NULL)), -EINVAL);

	/* Copy tbl24 entry */
	ptbl = (const uint32_t *)(&lpm->tbl24[tbl24_index]);
	tbl_entry = *ptbl;

	/* Memory ordering is not required in lookup. Because dataflow
	 * dependency exists, compiler or HW won't be able to re-order
	 * the operations.
	 */
	/* Copy tbl8 entry (only if needed) */
	if (unlikely((tbl_entry & RTE_LPM_VALID_EXT_ENTRY_BITMASK) ==
			RTE_LPM_VALID_EXT_ENTRY_BITMASK)) {

		unsigned tbl8_index = (uint8_t)ip +
				(((uint32_t)tbl_entry & 0x00FFFFFF) *
						RTE_LPM_TBL8_GROUP_NUM_ENTRIES);

		ptbl = (const uint32_t *)&lpm->tbl8[tbl8_index];
		tbl_entry = *ptbl;
	}

	*next_hop = ((uint32_t)tbl_entry & 0x00FFFFFF);
	return (tbl_entry & RTE_LPM_LOOKUP_SUCCESS) ? 0 : -ENOENT;
}

/**
 * Lookup multiple IP addresses in an LPM table. This may be implemented as a
 * macro, so the address of the function should not be used.
 *
 * @param lpm
 *   LPM object handle
 * @param ips
 *   Array of IPs to be looked up in the LPM table
 * @param next_hops
 *   Next hop of the most specific rule found for IP (valid on lookup hit only).
 *   This is an array of two byte values. The most significant byte in each
 *   value says whether the lookup was successful (bitmask
 *   RTE_LPM_LOOKUP_SUCCESS is set). The least significant byte is the
 *   actual next hop.
 * @param n
 *   Number of elements in ips (and next_hops) array to lookup. This should be a
 *   compile time constant, and divisible by 8 for best performance.
 *  @return
 *   -EINVAL for incorrect arguments, otherwise 0
 */
#define rte_lpm_lookup_bulk(lpm, ips, next_hops, n) \
		rte_lpm_lookup_bulk_func(lpm, ips, next_hops, n)

static inline int
rte_lpm_lookup_bulk_func(const struct rte_lpm *lpm, const uint32_t *ips,
		uint32_t *next_hops, const unsigned n)
{
	unsigned i;
	const uint32_t *ptbl;

	/* DEBUG: Check user input arguments. */
	RTE_LPM_RETURN_IF_TRUE(((lpm == NULL) || (ips == NULL) ||
			(next_hops == NULL)), -EINVAL);

	for (i = 0; i < n; i++) {
		unsigned int tbl24_index = ips[i] >> 8;

		/* Simply copy tbl24 entry to output */
		ptbl = (const uint32_t *)&lpm->tbl24[tbl24_index];
		next_hops[i] = *ptbl;

		/* Overwrite output with tbl8 entry if needed */
		if (unlikely((next_hops[i] & RTE_LPM_VALID_EXT_ENTRY_BITMASK) ==
				RTE_LPM_VALID_EXT_ENTRY_BITMASK)) {

			unsigned tbl8_index = (uint8_t)ips[i] +
					(((uint32_t)next_hops[i] & 0x00FFFFFF) *
					 RTE_LPM_TBL8_GROUP_NUM_ENTRIES);

			ptbl = (const uint32_t *)&lpm->tbl8[tbl8_index];
			next_hops[i] = *ptbl;
		}
	}
	return 0;
}

/* Mask four results. */
#define	 RTE_LPM_MASKX4_RES	UINT64_C(0x00ffffff00ffffff)

/**
 * Lookup four IP addresses in an LPM table.
 *
 * @param lpm
 *   LPM object handle
 * @param ip
 *   Four IPs to be looked up in the LPM table
 * @param hop
 *   Next hop of the most specific rule found for IP (valid on lookup hit only).
 *   This is an 4 elements array of two byte values.
 *   If the lookup was successful for the given IP, then least significant byte
 *   of the corresponding element is the  actual next hop and the most
 *   significant byte is zero.
 *   If the lookup for the given IP failed, then corresponding element would
 *   contain default value, see description of then next parameter.
 * @param defv
 *   Default value to populate into corresponding element of hop[] array,
 *   if lookup would fail.
 */
static inline void
rte_lpm_lookupx4(const struct rte_lpm *lpm, xmm_t ip, uint32_t hop[4],
	uint32_t defv);

#ifdef __cplusplus
}
#endif

#if defined(RTE_ARCH_ARM)
#ifdef RTE_HAS_SVE_ACLE
#include "rte_lpm_sve.h"
#undef rte_lpm_lookup_bulk
#define rte_lpm_lookup_bulk(lpm, ips, next_hops, n) \
		__rte_lpm_lookup_vec(lpm, ips, next_hops, n)
#endif
#include "rte_lpm_neon.h"
#elif defined(RTE_ARCH_PPC_64)
#include "rte_lpm_altivec.h"
#elif defined(RTE_ARCH_X86)
#include "rte_lpm_sse.h"
#elif defined(RTE_ARCH_RISCV) && defined(RTE_RISCV_FEATURE_V)
#include "rte_lpm_rvv.h"
#else
#include "rte_lpm_scalar.h"
#endif

#endif /* _RTE_LPM_H_ */
