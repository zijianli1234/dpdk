#include "acl_run.h"
#include <riscv_vector.h>
#include <stddef.h>
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
    for (i = 0; i < categories; i = i + vl) {
        // 设置向量长度，每次处理最多categories-i个元素
        vl = __riscv_vsetvl_e32m1(categories - i); // 使用m1，可以处理多个int32，vl为实际处理的元素个数

        // 加载当前匹配结果（规则ID和优先级）
        vuint32m1_t current_results = __riscv_vle32_v_u32m1(&p[transition].results[i], vl);
        vuint32m1_t current_priority = __riscv_vle32_v_u32m1(&p[transition].priority[i], vl);

        // 如果当前不是第一个完成的trie，需要与之前保存的结果比较
        if (parms[n].cmplt->count != ctx->num_tries) {
            // 加载之前保存的结果
            vuint32m1_t saved_results = __riscv_vle32_v_u32m1(&parms[n].cmplt->results[i], vl);
            vuint32m1_t saved_priority = __riscv_vle32_v_u32m1(&parms[n].cmplt->priority[i], vl);

            // 比较优先级：当前优先级是否小于已保存的优先级？注意：数值越小优先级越高
            vbool8_t mask = __riscv_vmslt_vv_u32m1_b8(current_priority, saved_priority, vl);

            // 混合规则ID：如果mask为真，则选择current_results，否则选择saved_results
            vuint32m1_t mixed_results = __riscv_vmerge_vvm_u32m1(mask, saved_results, current_results, vl);
            // 混合优先级
            vuint32m1_t mixed_priority = __riscv_vmerge_vvm_u32m1(mask, saved_priority, current_priority, vl);

            // 将混合后的结果存回
            __riscv_vse32_v_u32m1(&parms[n].cmplt->results[i], mixed_results, vl);
            __riscv_vse32_v_u32m1(&parms[n].cmplt->priority[i], mixed_priority, vl);
        } else {
            // 如果是第一个完成的trie，则直接保存当前结果
            __riscv_vse32_v_u32m1(&parms[n].cmplt->results[i], current_results, vl);
            __riscv_vse32_v_u32m1(&parms[n].cmplt->priority[i], current_priority, vl);
        }
    }
}

/*
 * Extract transitions from an XMM register and check for any matches
 */
// static void
// acl_process_matches(xmm_t *indices, int slot, const struct rte_acl_ctx *ctx,
// 	struct parms *parms, struct acl_flow_data *flows)
// {
// 	uint64_t transition1, transition2;

// 	/* extract transition from low 64 bits. */
// 	transition1 = _mm_cvtsi128_si64(*indices);

// 	/* extract transition from high 64 bits. */
// 	*indices = _mm_shuffle_epi32(*indices, SHUFFLE32_SWAP64);
// 	transition2 = _mm_cvtsi128_si64(*indices);

// 	transition1 = acl_match_check(transition1, slot, ctx,
// 		parms, flows, resolve_priority_sse);
// 	transition2 = acl_match_check(transition2, slot + 1, ctx,
// 		parms, flows, resolve_priority_sse);

// 	/* update indices with new transitions. */
// 	*indices = _mm_set_epi64x(transition2, transition1);
// }

/*
 * Extract transitions from a vector register and check for any matches (RVV 1.1 version)
 */
static void
acl_process_matches(uint64_t *indices, int slot, const struct rte_acl_ctx *ctx,
                    struct parms *parms, struct acl_flow_data *flows)
{
    // 设置向量长度为2个元素（两个64位状态）
    size_t vl = __riscv_vsetvl_e64m1(2);
    
    // 加载两个64位状态索引到向量寄存器
    vuint64m1_t v_indices = __riscv_vle64_v_u64m1(indices, vl);
    
    // 提取两个状态索引到标量变量
    uint64_t transition1 = __riscv_vmv_x_s_u64m1_u64(v_indices);
    uint64_t transition2 = __riscv_vmv_x_s_u64m1_u64(v_indices);
    
    // 处理第一个状态（低64位）
    transition1 = acl_match_check(transition1, slot, ctx, parms, flows, 
                                 resolve_priority_rvv);  // 使用RVV版本解析函数
    
    // 处理第二个状态（高64位）
    transition2 = acl_match_check(transition2, slot + 1, ctx, parms, flows, 
                                 resolve_priority_rvv);
    
    // 创建新的向量存储更新后的状态
    vuint64m1_t v_updated = vundefined_u64m1();
    v_updated = vset_u64m1(v_updated, 0, transition1);
    v_updated = vset_u64m1(v_updated, 1, transition2);
    
    // 存回内存
    vse64_v_u64m1(indices, v_updated, vl);
}



/*
 * Execute trie traversal with 8 traversals in parallel
 */
static inline int
search_rvv_8(const struct rte_acl_ctx *ctx, const uint8_t **data,
	      uint32_t *results, uint32_t total_packets, uint32_t categories)
{

}

/*
 * Execute trie traversal with 4 traversals in parallel
 */
static inline int
search_rvv_4(const struct rte_acl_ctx *ctx, const uint8_t **data,
	      uint32_t *results, int total_packets, uint32_t categories)
{

}
