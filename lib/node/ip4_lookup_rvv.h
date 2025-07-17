#ifndef __INCLUDE_IP4_LOOKUP_RVV_H__
#define __INCLUDE_IP4_LOOKUP_RVV_H__
/* RISC-V RVV */
static uint16_t
ip4_lookup_node_process_vec(struct rte_graph *graph, struct rte_node *node,
            void **objs, uint16_t nb_objs)
{
    struct rte_mbuf *mbuf0, *mbuf1, *mbuf2, *mbuf3, **pkts;
    struct rte_lpm *lpm = IP4_LOOKUP_NODE_LPM(node->ctx);
    const int dyn = IP4_LOOKUP_NODE_PRIV1_OFF(node->ctx);
    rte_edge_t next0, next1, next2, next3, next_index;
    struct rte_ipv4_hdr *ipv4_hdr;
    uint32_t ip0, ip1, ip2, ip3;
    void **to_next, **from;
    uint16_t last_spec = 0;
    uint16_t n_left_from;
    uint16_t held = 0;
    uint32_t drop_nh;
    uint32_t res[4]; // 存储lookupx4的结果
    // uint32_t ip_array[4]; // 存储IP地址的数组
    uint8_t bswap_idx[] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};
    size_t vl;
    int rc, i;
    rte_xmm_t ip_vec;

    /* Speculative next */
    next_index = RTE_NODE_IP4_LOOKUP_NEXT_REWRITE;
    /* Drop node */
    drop_nh = ((uint32_t)RTE_NODE_IP4_LOOKUP_NEXT_PKT_DROP) << 16;

    pkts = (struct rte_mbuf **)objs;
    from = objs;
    n_left_from = nb_objs;

    if (n_left_from >= 4) {
        for (i = 0; i < 4; i++)
            rte_prefetch0(rte_pktmbuf_mtod_offset(pkts[i], void *,
                        sizeof(struct rte_ether_hdr)));
    }

    /* Get stream for the speculated next node */
    to_next = rte_node_next_stream_get(graph, node, next_index, nb_objs);
    while (n_left_from >= 4) {
        /* Prefetch next-next mbufs */
        if (likely(n_left_from > 11)) {
            rte_prefetch0(pkts[8]);
            rte_prefetch0(pkts[9]);
            rte_prefetch0(pkts[10]);
            rte_prefetch0(pkts[11]);
        }

        /* Prefetch next mbuf data */
        if (likely(n_left_from > 7)) {
            rte_prefetch0(rte_pktmbuf_mtod_offset(pkts[4], void *,
                        sizeof(struct rte_ether_hdr)));
            rte_prefetch0(rte_pktmbuf_mtod_offset(pkts[5], void *,
                        sizeof(struct rte_ether_hdr)));
            rte_prefetch0(rte_pktmbuf_mtod_offset(pkts[6], void *,
                        sizeof(struct rte_ether_hdr)));
            rte_prefetch0(rte_pktmbuf_mtod_offset(pkts[7], void *,
                        sizeof(struct rte_ether_hdr)));
        }

        mbuf0 = pkts[0];
        mbuf1 = pkts[1];
        mbuf2 = pkts[2];
        mbuf3 = pkts[3];

        pkts += 4;
        n_left_from -= 4;

        /* Extract DIP and metadata for mbuf0 */
        ipv4_hdr = rte_pktmbuf_mtod_offset(mbuf0, struct rte_ipv4_hdr *,
                        sizeof(struct rte_ether_hdr));
        ip0 = ipv4_hdr->dst_addr;
        node_mbuf_priv1(mbuf0, dyn)->cksum = ipv4_hdr->hdr_checksum;
        node_mbuf_priv1(mbuf0, dyn)->ttl = ipv4_hdr->time_to_live;

        /* Extract DIP and metadata for mbuf1 */
        ipv4_hdr = rte_pktmbuf_mtod_offset(mbuf1, struct rte_ipv4_hdr *,
                        sizeof(struct rte_ether_hdr));
        ip1 = ipv4_hdr->dst_addr;
        node_mbuf_priv1(mbuf1, dyn)->cksum = ipv4_hdr->hdr_checksum;
        node_mbuf_priv1(mbuf1, dyn)->ttl = ipv4_hdr->time_to_live;

        /* Extract DIP and metadata for mbuf2 */
        ipv4_hdr = rte_pktmbuf_mtod_offset(mbuf2, struct rte_ipv4_hdr *,
                        sizeof(struct rte_ether_hdr));
        ip2 = ipv4_hdr->dst_addr;
        node_mbuf_priv1(mbuf2, dyn)->cksum = ipv4_hdr->hdr_checksum;
        node_mbuf_priv1(mbuf2, dyn)->ttl = ipv4_hdr->time_to_live;

        /* Extract DIP and metadata for mbuf3 */
        ipv4_hdr = rte_pktmbuf_mtod_offset(mbuf3, struct rte_ipv4_hdr *,
                        sizeof(struct rte_ether_hdr));
        ip3 = ipv4_hdr->dst_addr;
        node_mbuf_priv1(mbuf3, dyn)->cksum = ipv4_hdr->hdr_checksum;
        node_mbuf_priv1(mbuf3, dyn)->ttl = ipv4_hdr->time_to_live;

        /* Store IPs in array for vector processing */
        ip_vec.u32[0] = ip0;
        ip_vec.u32[1] = ip1;
        ip_vec.u32[2] = ip2;
        ip_vec.u32[3] = ip3;

        /* Vectorized byte swap using RVV */
        vl = __riscv_vsetvl_e32m1(4);
        vuint32m1_t v_ip = __riscv_vle32_v_u32m1(ip_vec.u32, vl);
        vuint8m1_t v_ip_bytes = __riscv_vreinterpret_v_u32m1_u8m1(v_ip);
        vuint8m1_t v_idx = __riscv_vle8_v_u8m1(bswap_idx, 16);
        vuint8m1_t v_bswap = __riscv_vrgather_vv_u8m1(v_ip_bytes, v_idx, vl * 4);
        v_ip = __riscv_vreinterpret_v_u8m1_u32m1(v_bswap);
        __riscv_vse32_v_u32m1(ip_vec.u32, v_ip, vl);

        /* Perform LPM lookup */
        rte_lpm_lookupx4(lpm, ip_vec.x, res, drop_nh);

        /* Update statistics */
        for (i = 0; i < 4; i++) {
            if ((res[i] >> 16) == (drop_nh >> 16)) {
                NODE_INCREMENT_XSTAT_ID(node, 0, 1, 1);
            }
        }

        /* Extract next hop and next node */
        node_mbuf_priv1(mbuf0, dyn)->nh = res[0] & 0xFFFF;
        next0 = res[0] >> 16;

        node_mbuf_priv1(mbuf1, dyn)->nh = res[1] & 0xFFFF;
        next1 = res[1] >> 16;

        node_mbuf_priv1(mbuf2, dyn)->nh = res[2] & 0xFFFF;
        next2 = res[2] >> 16;

        node_mbuf_priv1(mbuf3, dyn)->nh = res[3] & 0xFFFF;
        next3 = res[3] >> 16;

        /* Enqueue four to next node */
        rte_edge_t fix_spec =
            (next_index ^ next0) | (next_index ^ next1) |
            (next_index ^ next2) | (next_index ^ next3);

        if (unlikely(fix_spec)) {
            /* Copy successfully speculated packets */
            rte_memcpy(to_next, from, last_spec * sizeof(from[0]));
            from += last_spec;
            to_next += last_spec;
            held += last_spec;
            last_spec = 0;

            /* Process each packet individually */
            if (next_index == next0) {
                *to_next++ = from[0];
                held++;
            } else {
                rte_node_enqueue_x1(graph, node, next0, from[0]);
            }

            if (next_index == next1) {
                *to_next++ = from[1];
                held++;
            } else {
                rte_node_enqueue_x1(graph, node, next1, from[1]);
            }

            if (next_index == next2) {
                *to_next++ = from[2];
                held++;
            } else {
                rte_node_enqueue_x1(graph, node, next2, from[2]);
            }

            if (next_index == next3) {
                *to_next++ = from[3];
                held++;
            } else {
                rte_node_enqueue_x1(graph, node, next3, from[3]);
            }

            from += 4;
        } else {
            last_spec += 4;
        }
    }

    /* Process remaining packets */
    while (n_left_from > 0) {
        uint32_t next_hop;

        mbuf0 = pkts[0];
        pkts += 1;
        n_left_from -= 1;

        ipv4_hdr = rte_pktmbuf_mtod_offset(mbuf0, struct rte_ipv4_hdr *,
                        sizeof(struct rte_ether_hdr));
        node_mbuf_priv1(mbuf0, dyn)->cksum = ipv4_hdr->hdr_checksum;
        node_mbuf_priv1(mbuf0, dyn)->ttl = ipv4_hdr->time_to_live;

        rc = rte_lpm_lookup(lpm, rte_be_to_cpu_32(ipv4_hdr->dst_addr), &next_hop);
        next_hop = (rc == 0) ? next_hop : drop_nh;
        NODE_INCREMENT_XSTAT_ID(node, 0, rc != 0, 1);

        node_mbuf_priv1(mbuf0, dyn)->nh = next_hop & 0xFFFF;
        next0 = next_hop >> 16;

        if (unlikely(next_index != next0)) {
            rte_memcpy(to_next, from, last_spec * sizeof(from[0]));
            from += last_spec;
            to_next += last_spec;
            held += last_spec;
            last_spec = 0;
            rte_node_enqueue_x1(graph, node, next0, from[0]);
            from += 1;
        } else {
            last_spec += 1;
        }
    }

    /* Handle successfully speculated packets */
    if (likely(last_spec == nb_objs)) {
        rte_node_next_stream_move(graph, node, next_index);
        return nb_objs;
    }

    held += last_spec;
    rte_memcpy(to_next, from, last_spec * sizeof(from[0]));
    rte_node_next_stream_put(graph, node, next_index, held);

    return nb_objs;
}
#endif