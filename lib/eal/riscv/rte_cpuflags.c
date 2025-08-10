/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2022 StarFive
 * Copyright(c) 2022 SiFive
 * Copyright(c) 2022 Semihalf
 */

#include <eal_export.h>
#include "rte_cpuflags.h"

#include <elf.h>
#include <fcntl.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/syscall.h>

/*
 * when hardware probing is not possible, we assume all extensions are missing
 * at runtime
 */
#ifdef RTE_RISCV_FEATURE_HWPROBE
#include <asm/hwprobe.h>
#endif

#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif

#ifndef AT_HWCAP2
#define AT_HWCAP2 26
#endif

#ifndef AT_PLATFORM
#define AT_PLATFORM 15
#endif

enum cpu_register_t {
	REG_NONE = 0,
	REG_HWCAP,
	REG_HWCAP2,
	REG_PLATFORM,
	REG_HWPROBE_IMA_EXT_0,
	REG_MAX,
};

typedef uint32_t hwcap_registers_t[REG_MAX];

/**
 * Struct to hold a processor feature entry
 */
struct feature_entry {
	uint32_t reg;
	uint64_t mask;
#define CPU_FLAG_NAME_MAX_LEN 64
	char name[CPU_FLAG_NAME_MAX_LEN];
};

#define FEAT_DEF(name, reg, mask) \
	[RTE_CPUFLAG_##name] = {reg, mask, #name},

typedef Elf64_auxv_t _Elfx_auxv_t;

const struct feature_entry rte_cpu_feature_table[] = {
	FEAT_DEF(RISCV_ISA_A, REG_HWCAP, 1 <<  0)
	FEAT_DEF(RISCV_ISA_B, REG_HWCAP, 1 <<  1)
	FEAT_DEF(RISCV_ISA_C, REG_HWCAP, 1 <<  2)
	FEAT_DEF(RISCV_ISA_D, REG_HWCAP, 1 <<  3)
	FEAT_DEF(RISCV_ISA_E, REG_HWCAP, 1 <<  4)
	FEAT_DEF(RISCV_ISA_F, REG_HWCAP, 1 <<  5)
	FEAT_DEF(RISCV_ISA_G, REG_HWCAP, 1 <<  6)
	FEAT_DEF(RISCV_ISA_H, REG_HWCAP, 1 <<  7)
	FEAT_DEF(RISCV_ISA_I, REG_HWCAP, 1 <<  8)
	FEAT_DEF(RISCV_ISA_J, REG_HWCAP, 1 <<  9)
	FEAT_DEF(RISCV_ISA_K, REG_HWCAP, 1 << 10)
	FEAT_DEF(RISCV_ISA_L, REG_HWCAP, 1 << 11)
	FEAT_DEF(RISCV_ISA_M, REG_HWCAP, 1 << 12)
	FEAT_DEF(RISCV_ISA_N, REG_HWCAP, 1 << 13)
	FEAT_DEF(RISCV_ISA_O, REG_HWCAP, 1 << 14)
	FEAT_DEF(RISCV_ISA_P, REG_HWCAP, 1 << 15)
	FEAT_DEF(RISCV_ISA_Q, REG_HWCAP, 1 << 16)
	FEAT_DEF(RISCV_ISA_R, REG_HWCAP, 1 << 17)
	FEAT_DEF(RISCV_ISA_S, REG_HWCAP, 1 << 18)
	FEAT_DEF(RISCV_ISA_T, REG_HWCAP, 1 << 19)
	FEAT_DEF(RISCV_ISA_U, REG_HWCAP, 1 << 20)
	FEAT_DEF(RISCV_ISA_V, REG_HWCAP, 1 << 21)
	FEAT_DEF(RISCV_ISA_W, REG_HWCAP, 1 << 22)
	FEAT_DEF(RISCV_ISA_X, REG_HWCAP, 1 << 23)
	FEAT_DEF(RISCV_ISA_Y, REG_HWCAP, 1 << 24)
	FEAT_DEF(RISCV_ISA_Z, REG_HWCAP, 1 << 25)

#ifdef RTE_RISCV_FEATURE_ZBC
	FEAT_DEF(RISCV_EXT_ZBC, REG_HWPROBE_IMA_EXT_0, RISCV_HWPROBE_EXT_ZBC)
#else
	FEAT_DEF(RISCV_EXT_ZBC, REG_HWPROBE_IMA_EXT_0, 0)
#endif
};

#ifdef RTE_RISCV_FEATURE_HWPROBE
/*
 * Use kernel interface for probing hardware capabilities to get extensions
 * present on this machine
 */
static uint64_t
rte_cpu_hwprobe_ima_ext(void)
{
	printf("rte_cpu_hwprobe_ima_ext enter \n");
	long ret;
	struct riscv_hwprobe extensions_pair;

	struct riscv_hwprobe *pairs = &extensions_pair;
	size_t pair_count = 1;
	/* empty set of cpus returns extensions present on all cpus */
	cpu_set_t *cpus = NULL;
	size_t cpusetsize = 0;
	unsigned int flags = 0;

	extensions_pair.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
	ret = syscall(__NR_riscv_hwprobe, pairs, pair_count, cpusetsize, cpus,
		      flags);

	if (ret != 0)
		return 0;
	return extensions_pair.value;
}
#endif /* RTE_RISCV_FEATURE_HWPROBE */
/*
 * Read AUXV software register and get cpu features for ARM
 */
static void
rte_cpu_get_features(hwcap_registers_t out)
{
	out[REG_HWCAP] = rte_cpu_getauxval(AT_HWCAP);
	out[REG_HWCAP2] = rte_cpu_getauxval(AT_HWCAP2);
#ifdef RTE_RISCV_FEATURE_HWPROBE
	out[REG_HWPROBE_IMA_EXT_0] = rte_cpu_hwprobe_ima_ext();
#endif
}

/*
 * Checks if a particular flag is available on current machine.
 */
RTE_EXPORT_SYMBOL(rte_cpu_get_flag_enabled)
int
rte_cpu_get_flag_enabled(enum rte_cpu_flag_t feature)
{
	const struct feature_entry *feat;
	hwcap_registers_t regs = {0};

	if ((unsigned int)feature >= RTE_DIM(rte_cpu_feature_table))
		return -ENOENT;

	feat = &rte_cpu_feature_table[feature];
	if (feat->reg == REG_NONE)
		return -EFAULT;

	rte_cpu_get_features(regs);
	return (regs[feat->reg] & feat->mask) != 0;
}

RTE_EXPORT_SYMBOL(rte_cpu_get_flag_name)
const char *
rte_cpu_get_flag_name(enum rte_cpu_flag_t feature)
{
	if ((unsigned int)feature >= RTE_DIM(rte_cpu_feature_table))
		return NULL;
	return rte_cpu_feature_table[feature].name;
}

RTE_EXPORT_SYMBOL(rte_cpu_get_intrinsics_support)
void
rte_cpu_get_intrinsics_support(struct rte_cpu_intrinsics *intrinsics)
{
	memset(intrinsics, 0, sizeof(*intrinsics));
}
