#include "acl_run.h"
#include "rte_vect.h"
#include <glob.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static const rte_xmm_t xmm_match_mask = {
	.u32 = {
		RTE_ACL_NODE_MATCH,
		RTE_ACL_NODE_MATCH,
		RTE_ACL_NODE_MATCH,
		RTE_ACL_NODE_MATCH,
	},
};


static const rte_xmm_t xmm_index_mask = {
	.u32 = {
		RTE_ACL_NODE_INDEX,
		RTE_ACL_NODE_INDEX,
		RTE_ACL_NODE_INDEX,
		RTE_ACL_NODE_INDEX,
	},
};
void print_rvv_vector_bytes(const char* name, vint32m1_t vec, size_t vl) {
    uint8_t bytes[16];
    vint8m1_t byte_vec = __riscv_vreinterpret_v_i32m1_i8m1(vec);
    __riscv_vse8_v_i8m1(bytes, byte_vec, vl * 4);
    
    printf("%-10s: ", name);
    for (int i = 0; i < vl * 4; i++) {
        printf("%02x ", bytes[i]);
        if ((i + 1) % 4 == 0) printf(" "); // 每4字节加空格
    }
    printf("\n");
}

// 打印vbool32_t掩码
static void print_vbool32_t(const char *name, vbool32_t mask, size_t vl) {
    uint8_t buffer[1]; // 由于vl=4，只需要1个字节（因为每8个元素才占1个字节？不对，这里vbool32_t是每32位一个掩码位）
    // 实际上，vbool32_t的存储方式是每个元素1位，所以4个元素需要4位，即0.5字节。但RVV提供vsm来存储掩码，存储空间为(vl+7)/8字节
    size_t byte_size = (vl + 7) / 8;
    uint8_t bytes[byte_size];
    __riscv_vsm_v_b32(bytes, mask, vl);
    printf("%s: ", name);
    for (size_t i = 0; i < byte_size; i++) {
        for (int j = 0; j < 8; j++) {
            if (i*8+j < vl) {
                printf("%d ", (bytes[i] >> j) & 1);
            }
        }
    }
    printf("\n");
}

/*
 * Resolve priority for multiple results (rvv version).
 * This consists comparing the priority of the current traversal with the
 * running set of results for the packet.
 * For each result, keep a running array of the result (rule number) and
 * its priority for each category.
 */
// transition:当前匹配到的节点在结果数组中的索引
// n:当前处理的数据包索引
// ctx:ACL 上下文（含 trie 树数量等）
// parms:数据包匹配状态数组
// p:	全局匹配结果表
// categories:分类维度数（如安全/QoS 等）
static inline void
resolve_priority_rvv(uint64_t transition, int n, const struct rte_acl_ctx *ctx,
                     struct parms *parms, const struct rte_acl_match_results *p,
                     uint32_t categories)
{
    size_t i;
    size_t vl = 0;
    for (i = 0; i < categories; i += 4) {
        vl = __riscv_vsetvl_e32m1(4);
        // 加载当前匹配结果（规则ID和优先级）
        vuint32m1_t vu32_current_results = __riscv_vle32_v_u32m1(&p[transition].results[i], vl);
        vint32m1_t vi32_current_priority = __riscv_vle32_v_i32m1(&p[transition].priority[i], vl);

        if (parms[n].cmplt->count != ctx->num_tries) {
            // 加载之前保存的结果
            vuint32m1_t vu32_saved_results = __riscv_vle32_v_u32m1(&parms[n].cmplt->results[i], vl);
            vint32m1_t vi32_saved_priority = __riscv_vle32_v_i32m1(&parms[n].cmplt->priority[i], vl);
            // 比较优先级：当前优先级是否小于已保存_的优先级？注意：数值越小优先级越高
            vbool32_t vb32_mask = __riscv_vmslt_vv_i32m1_b32(vi32_current_priority, vi32_saved_priority, vl);
            vuint32m1_t vu32_mixed_results = __riscv_vmerge_vvm_u32m1(vu32_current_results, vu32_saved_results, vb32_mask, vl);

            // 混合优先级
            // vint32m1_t vi32_mixed_priority = __riscv_vmerge_vvm_i32m1(vi32_saved_priority, vi32_current_priority, vb32_mask, vl);
            vint32m1_t vi32_mixed_priority = __riscv_vmerge_vvm_i32m1(vi32_current_priority, vi32_saved_priority, vb32_mask, vl);

            vu32_current_results = vu32_mixed_results;
            vi32_current_priority = vi32_mixed_priority;

        } 
            // 如果是第一个完成的trie，则直接保存当前结果
            __riscv_vse32_v_u32m1(&parms[n].cmplt->results[i], vu32_current_results, vl);
            __riscv_vse32_v_i32m1(&parms[n].cmplt->priority[i], vi32_current_priority, vl);
    }
}

/*
 * Extract transitions from a vector register and check for any matches (RVV 1.1 version)
 */
static void
acl_process_matches(uint64_t *u64_indices, int slot, const struct rte_acl_ctx *ctx,
	struct parms *parms, struct acl_flow_data *flows)
{
    u64_indices[0] = acl_match_check(u64_indices[0], slot, ctx, parms, flows, 
                                 resolve_priority_rvv);  // 使用RVV版本解析函数
    // printf("acl_process_matches() start:%d!\n", __LINE__);
    // 处理第二个状态（高64位）
    u64_indices[1] = acl_match_check(u64_indices[1], slot + 1, ctx, parms, flows, 
                                 resolve_priority_rvv);
}

// 输入：128 位向量（4 个 32 位元素）
// 输出：每个 32 位元素的字节求和结果（32 位）
static inline vint32m1_t rvv_quad_calc(vint32m1_t input, size_t vl) {
    // 1. 提取每个元素的低 16 位（包含 2 个字节）
    vuint32m1_t input2 = __riscv_vreinterpret_v_i32m1_u32m1(input);
    vuint32m1_t low16 = __riscv_vand_vx_u32m1(input2, 0xFFFF, vl);
    // // print_rvv_vector_bytes("low16", low16, 4);
    // 2. 提取每个元素的高 16 位（包含 2 个字节）
    vuint32m1_t high16 = __riscv_vsrl_vx_u32m1(input2, 16, vl);
    // // print_rvv_vector_bytes("high16", high16, 4);
    // 3. 将每个 16 位部分拆分为高低字节并求和
    vuint32m1_t sum_low = __riscv_vadd_vv_u32m1(
        __riscv_vand_vx_u32m1(low16, 0xFF, vl),      // 低字节
        __riscv_vsrl_vx_u32m1(low16, 8, vl),         // 高字节
        vl
    );
    // // print_rvv_vector_bytes("sum_low", sum_low, 4);
    vuint32m1_t sum_high = __riscv_vadd_vv_u32m1(
        __riscv_vand_vx_u32m1(high16, 0xFF, vl),     // 低字节
        __riscv_vsrl_vx_u32m1(high16, 8, vl),        // 高字节
        vl
    );
    // // print_rvv_vector_bytes("sum_high", sum_high, 4);
    // 4. 合并低 16 位和高 16 位的字节和
    return __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vadd_vv_u32m1(sum_low, sum_high, vl));
}

static inline __attribute__((always_inline)) vint32m1_t
calc_addr_rvv(uint32_t index_mask, vint32m1_t next_input,
     vint32m1_t tr_lo, vint32m1_t tr_hi)
{
    size_t vl = 4;
    const int32_t range_base[4] = {0xffffff00, 0xffffff04, 0xffffff08, 0xffffff0c};

    vint32m1_t vi32_t = __riscv_vmv_v_x_i32m1(0, 4);

    vint32m1_t low8 = __riscv_vand_vx_i32m1(next_input, 0xFF, 4);
    vint32m1_t vi32_in = __riscv_vmul_vx_i32m1(low8, 0x01010101, vl);
    vint32m1_t vi32_node_type = __riscv_vand_vx_i32m1(tr_lo,
        ~index_mask , vl);
    vint32m1_t vi32_addr = __riscv_vand_vx_i32m1(tr_lo, index_mask, vl);
    vbool32_t vb32_dfa_msk = __riscv_vmseq_vx_i32m1_b32(vi32_node_type, 0, vl);
    vint32m1_t vi32_r = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vsrl_vx_u32m1(__riscv_vreinterpret_v_i32m1_u32m1(vi32_in), 30, vl));

    vi32_r = __riscv_vadd_vv_i32m1(vi32_r, __riscv_vle32_v_i32m1(range_base, vl), vl);

    vuint8m1_t vu8_r = __riscv_vreinterpret_v_i8m1_u8m1(__riscv_vreinterpret_v_i32m1_i8m1(vi32_r));
    vint8m1_t v_tr_hi8 = __riscv_vreinterpret_v_i32m1_i8m1(tr_hi);
    vint8m1_t vi8_res = __riscv_vrgather_vv_i8m1(v_tr_hi8, vu8_r, 16);

    vi32_r = __riscv_vreinterpret_v_i8m1_i32m1(vi8_res);

    vi32_t = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vsrl_vx_u32m1(__riscv_vreinterpret_v_i32m1_u32m1(vi32_in), 24, vl));
    vint32m1_t vi32_dfa_ofs = __riscv_vsub_vv_i32m1(vi32_t, vi32_r, vl);
    vbool8_t compare = __riscv_vmsgt_vv_i8m1_b8(__riscv_vreinterpret_v_i32m1_i8m1((vi32_in)),
                         __riscv_vreinterpret_v_i32m1_i8m1((tr_hi)), vl*4);
    vint8m1_t mask_int = __riscv_vmerge_vxm_i8m1(__riscv_vmv_v_x_i8m1(0, vl*4),
                                         1,
                                         compare,
                                         vl*4);
        
    vint32m1_t vi32_quad_ofs = rvv_quad_calc(__riscv_vreinterpret_v_i8m1_i32m1(mask_int), 4);
    vint32m1_t vi32_offset = __riscv_vmerge_vvm_i32m1(vi32_quad_ofs, vi32_dfa_ofs, vb32_dfa_msk, vl);
    return __riscv_vadd_vv_i32m1(vi32_addr, vi32_offset, vl);
}

/*
 * Process 4 transitions (in 2 XMM registers) in parallel
 */
static __rte_always_inline vint32m1_t
transition4(vint32m1_t vi32_next_input, const uint64_t *trans,
	uint64_t *u64p_indices1, uint64_t *u64p_indices2)
{
    vint32m1_t vi32_tr_lo, vi32_tr_hi;
    vint32m1_t vi32_addr;
    uint64_t trans1[4] = {0};
    size_t vl = 4;
    // 分离低32位和高32位
	/* Shuffle low 32 into vi32_tr_lo and high 32 into vi32_tr_hi */
    vint64m2_t vi64m2_indices = __riscv_vle64_v_i64m2((int64_t*)u64p_indices1, 4);
    vi32_tr_lo = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(__riscv_vreinterpret_v_i64m2_u64m2(vi64m2_indices),  0, 4));
    vi32_tr_hi = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(__riscv_vreinterpret_v_i64m2_u64m2(vi64m2_indices), 32, 4));
    vi32_addr = calc_addr_rvv(RTE_ACL_NODE_INDEX, vi32_next_input, vi32_tr_lo, vi32_tr_hi);

    u64p_indices1[0] = trans[__riscv_vmv_x_s_i32m1_i32(vi32_addr)];
	/* get slot 2 */
        
	/* {x0, x1, x2, x3} -> {x2, x1, x2, x3} */
    vint32m1_t vi32_temp = __riscv_vslidedown_vx_i32m1(vi32_addr, 2, vl);
    u64p_indices2[0] = trans[__riscv_vmv_x_s_i32m1_i32(vi32_temp)];

	/* get slot 1 */

	/* {x2, x1, x2, x3} -> {x1, x1, x2, x3} */
    vi32_temp = __riscv_vslidedown_vx_i32m1(vi32_addr, 1, vl);
    u64p_indices1[1] = trans[__riscv_vmv_x_s_i32m1_i32(vi32_temp)];
	/* get slot 3 */

	/* {x1, x1, x2, x3} -> {x3, x1, x2, x3} */
    vi32_temp = __riscv_vslidedown_vx_i32m1(vi32_addr, 3, vl);
    u64p_indices2[1] = trans[__riscv_vmv_x_s_i32m1_i32(vi32_temp)];

    return __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vsrl_vx_u32m1(__riscv_vreinterpret_v_i32m1_u32m1(vi32_next_input), CHAR_BIT, 4));
}


/*
 * Check for any match in 4 transitions (contained in 2 RVV registers)
 */
static __rte_always_inline void
acl_match_check_x4(int slot, const struct rte_acl_ctx *ctx, struct parms *parms,
	struct acl_flow_data *flows, uint64_t *indices1, uint64_t *indices2,
	uint32_t match_mask)
{
    size_t vl = __riscv_vsetvl_e32m1(4);
    vbool32_t v_nonzero_mask;
    long first_nonzero;

    vint64m2_t vi64m2_indices64;
    vint32m1_t vi32_temp;

    // vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
    // printf("enter acl_match_check_x4:%d\n", __LINE__);
    while (1) {
        vi64m2_indices64 = __riscv_vle64_v_i64m2((int64_t*)indices1, vl);
        // vi32_temp = __riscv_vnsra_wx_i32m1(vi64m2_indices64, 0, 4);
        vi32_temp = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(__riscv_vreinterpret_v_i64m2_u64m2(vi64m2_indices64), 0, 4));


        vi32_temp = __riscv_vand_vx_i32m1(vi32_temp, match_mask, vl);
        v_nonzero_mask = __riscv_vmsne_vx_i32m1_b32(vi32_temp, 0, vl);
        first_nonzero = __riscv_vfirst_m_b32(v_nonzero_mask, vl);
        // 如果没有匹配则退出循环
        if (first_nonzero < 0) break;
        // 处理匹配结果
        acl_process_matches(indices1, slot, ctx, parms, flows);
        acl_process_matches(indices2, slot + 2, ctx, parms, flows);
        // print_rvv_vector_bytes("vi32_temp", vi32_temp, 4);
    }
    // __riscv_vmerge_vvm_i32m1();
}

/*
 * Execute trie traversal with 8 traversals in parallel
 */
static inline int
search_rvv_8(const struct rte_acl_ctx *ctx, const uint8_t **data,
	      uint32_t *results, uint32_t total_packets, uint32_t categories)
{
    int n;
	struct acl_flow_data flows;
	uint64_t index_array[MAX_SEARCHES_RVV8];
	struct completion cmplt[MAX_SEARCHES_RVV8];
	struct parms parms[MAX_SEARCHES_RVV8];
    vint32m1_t vi32_input0, vi32_input1;

	acl_set_flow(&flows, cmplt, RTE_DIM(cmplt), data, results,
		total_packets, categories, ctx->trans_table);

	for (n = 0; n < MAX_SEARCHES_RVV8; n++)
		index_array[n] = acl_start_next_trie(&flows, parms, n, ctx);

    // 转换为RVV代码

    vint64m1_t vu64_indices1 = __riscv_vle64_v_i64m1(&index_array[0], 2);
    vint64m1_t vu64_indices2 = __riscv_vle64_v_i64m1(&index_array[2], 2);
    vint64m1_t vu64_indices3 = __riscv_vle64_v_i64m1(&index_array[4], 2);
    vint64m1_t vu64_indices4 = __riscv_vle64_v_i64m1(&index_array[6], 2);

    	 /* Check for any matches. */
	acl_match_check_x4(0, ctx, parms, &flows,
		&index_array[0], &index_array[2], RTE_ACL_NODE_MATCH);
    acl_match_check_x4(4, ctx, parms, &flows,
		&index_array[4], &index_array[6], RTE_ACL_NODE_MATCH);
    int j = 0;
    while (flows.started > 0) {

        int32_t input_data0[4] = {
            GET_NEXT_4BYTES(parms, 0),  // 流0 -> 最低位
            GET_NEXT_4BYTES(parms, 1),  // 流1
            GET_NEXT_4BYTES(parms, 2),  // 流2
            GET_NEXT_4BYTES(parms, 3)   // 流3 -> 最高位
        };
        int32_t input_data1[4] = {
            GET_NEXT_4BYTES(parms, 4),  // 流0 -> 最低位
            GET_NEXT_4BYTES(parms, 5),  // 流1
            GET_NEXT_4BYTES(parms, 6),  // 流2
            GET_NEXT_4BYTES(parms, 7)   // 流3 -> 最高位
        };


        vi32_input0 = __riscv_vle32_v_i32m1(&input_data0[0], 4);
        vi32_input1 = __riscv_vle32_v_i32m1(&input_data1[0], 4);
        
        /* Process the 4 bytes of input on each stream. */

		vi32_input0 = transition4(vi32_input0, flows.trans,
			&index_array[0], &index_array[2]);
		vi32_input1 = transition4(vi32_input1, flows.trans,
			&index_array[4], &index_array[6]);

		vi32_input0 = transition4(vi32_input0, flows.trans,
			&index_array[0], &index_array[2]);
		vi32_input1 = transition4(vi32_input1, flows.trans,
			&index_array[4], &index_array[6]);

		vi32_input0 = transition4(vi32_input0, flows.trans,
			&index_array[0], &index_array[2]);
		vi32_input1 = transition4(vi32_input1, flows.trans,
			&index_array[4], &index_array[6]);

		vi32_input0 = transition4(vi32_input0, flows.trans,
			&index_array[0], &index_array[2]);
		vi32_input1 = transition4(vi32_input1, flows.trans,
			&index_array[4], &index_array[6]);

		 /* Check for any matches. */
		acl_match_check_x4(0, ctx, parms, &flows,
			&index_array[0], &index_array[2], RTE_ACL_NODE_MATCH);
		acl_match_check_x4(4, ctx, parms, &flows,
			&index_array[4], &index_array[6], RTE_ACL_NODE_MATCH);
    }
    return 0;
}

/*
 * Execute trie traversal with 4 traversals in parallel
 */
static inline int
search_rvv_4(const struct rte_acl_ctx *ctx, const uint8_t **data,
	      uint32_t *results, int total_packets, uint32_t categories)
{
    
    int n;
	struct acl_flow_data flows;
	uint64_t index_array[MAX_SEARCHES_RVV4];

	struct completion cmplt[MAX_SEARCHES_RVV4];
	struct parms parms[MAX_SEARCHES_RVV4];
    int32_t input_0[4] = {0};
    vint32m1_t vu32_input0;
    acl_set_flow(&flows, cmplt, RTE_DIM(cmplt), data, results,
		total_packets, categories, ctx->trans_table);
    for (n = 0; n < MAX_SEARCHES_RVV4; n++)
		index_array[n] = acl_start_next_trie(&flows, parms, n, ctx);
    acl_match_check_x4(0, ctx, parms, &flows,
		&index_array[0], &index_array[2], RTE_ACL_NODE_MATCH);
    // printf("while loop start\n");
    int j = 0;
    while (flows.started > 0) {
        int32_t input_data[4] = {
            GET_NEXT_4BYTES(parms, 0),  // 流0 -> 最低位
            GET_NEXT_4BYTES(parms, 1),  // 流1
            GET_NEXT_4BYTES(parms, 2),  // 流2
            GET_NEXT_4BYTES(parms, 3)   // 流3 -> 最高位
        };

        vu32_input0 = __riscv_vle32_v_i32m1(&input_data[0], 4);
        vu32_input0 = transition4(vu32_input0, flows.trans,
             &index_array[0], &index_array[2]);
        
        vu32_input0 = transition4(vu32_input0, flows.trans,
			&index_array[0], &index_array[2]);

        vu32_input0 = transition4(vu32_input0, flows.trans,
			&index_array[0], &index_array[2]);

        vu32_input0 = transition4(vu32_input0, flows.trans,
			&index_array[0], &index_array[2]);

        /* Check for any matches. */
		acl_match_check_x4(0, ctx, parms, &flows,
			&index_array[0], &index_array[2], RTE_ACL_NODE_MATCH);
    }
    return 0;
}
