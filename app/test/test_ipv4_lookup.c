#include <rte_graph.h>
#include <rte_mbuf.h>
#include <rte_mbuf_pool_ops.h>
#include <rte_ip.h>
#include <rte_lpm.h>
// #include <rte_node_ip4.h>
#include <rte_node_ip4_api.h>
#include <rte_ethdev.h>
#include <cmdline_parse.h>
#include <cmdline_parse_etheraddr.h>
#include <cmdline_parse_ipaddr.h>
#include <cmdline_parse_string.h>
#include <rte_ether.h>
#include <rte_graph_worker_common.h>
#include <stddef.h>
#include "test.h"

/* Test routes */
static struct {
    uint32_t ip;
    uint8_t depth;
    uint8_t next_hop;
} test_routes[] = {
    {RTE_IPV4(10, 0, 0, 0), 24, 1},
    {RTE_IPV4(192, 168, 1, 0), 24, 2},
    {RTE_IPV4(172, 16, 0, 0), 16, 3},
};

/* Test packets */
static struct test_packet {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint8_t expected_next_hop;
} test_packets[] = {
    {RTE_IPV4(192, 168, 0, 1), RTE_IPV4(10, 0, 0, 5), 1},
    {RTE_IPV4(10, 0, 1, 1), RTE_IPV4(192, 168, 1, 100), 2},
    {RTE_IPV4(172, 17, 0, 1), RTE_IPV4(172, 16, 1, 1), 3},
    {RTE_IPV4(8, 8, 8, 8), RTE_IPV4(1, 1, 1, 1), RTE_NODE_IP4_LOOKUP_NEXT_PKT_DROP},
};

/* Global resources */
static struct rte_mempool *mbuf_pool;
static rte_graph_t test_graph_id;
static struct rte_graph* test_graph;
static struct rte_node *ip4_lookup_node;
static int next_hop_dynfield_offset = -1;

#define GET_NEXT_HOP(m) (*(RTE_MBUF_DYNFIELD((m), next_hop_dynfield_offset, uint64_t *)))

static int test_ip4_lookup_setup(void)
{
    int ret;
    const char *node_patterns[] = {"ip4_lookup"};
    struct rte_graph_param graph_conf = {
        .node_patterns = node_patterns,
        .nb_node_patterns = 1,
        .socket_id = SOCKET_ID_ANY
    };

    const struct rte_mbuf_dynfield dynfield_desc = {
        .name = "next_hop_dynfield",
        .size = sizeof(uint64_t),
        .align = __alignof__(uint64_t),
        .flags = 0,
    };
    next_hop_dynfield_offset = rte_mbuf_dynfield_register(&dynfield_desc);
    if (next_hop_dynfield_offset < 0) {
        printf("Failed to register dynamic field\n");
        return -1;
    }

    mbuf_pool = rte_pktmbuf_pool_create("test_pool", 128, 0,
                                        0, RTE_MBUF_DEFAULT_BUF_SIZE,
                                        SOCKET_ID_ANY);
    if (!mbuf_pool)
        return -1;

    test_graph_id = rte_graph_create("ip4_lookup_test", &graph_conf);
    if (!test_graph_id)
        return -1;
    // rte_node_t node = rte_node_from_name("ip4_lookup");
    ip4_lookup_node = rte_graph_node_get_by_name("ip4_lookup_test", "ip4_lookup");
    // ip4_lookup_node = rte_graph_node_get(test_graph, "ip4_lookup");
    if (!ip4_lookup_node)
        return -1;

    for (size_t i = 0; i < RTE_DIM(test_routes); i++) {
        ret = rte_node_ip4_route_add(
            test_routes[i].ip,
            test_routes[i].depth,
            test_routes[i].next_hop,
            RTE_NODE_IP4_LOOKUP_NEXT_REWRITE
        );
        if (ret < 0)
            return ret;
    }

    return 0;
}

static void test_ip4_lookup_teardown(void)
{
    if (test_graph_id)
        rte_graph_destroy(test_graph_id);
    if (mbuf_pool)
        rte_mempool_free(mbuf_pool);
}

static struct rte_mbuf *create_test_packet(uint32_t src_ip, uint32_t dst_ip)
{
    struct rte_mbuf *mbuf = rte_pktmbuf_alloc(mbuf_pool);
    if (!mbuf)
        return NULL;

    struct rte_ether_hdr *eth = (struct rte_ether_hdr *)rte_pktmbuf_append(mbuf, sizeof(struct rte_ether_hdr));
    struct rte_ipv4_hdr *ipv4 = (struct rte_ipv4_hdr *)rte_pktmbuf_append(mbuf, sizeof(struct rte_ipv4_hdr));

    memset(eth, 0, sizeof(*eth));
    eth->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

    ipv4->version_ihl = RTE_IPV4_VHL_DEF;
    ipv4->type_of_service = 0;
    ipv4->total_length = rte_cpu_to_be_16(sizeof(*ipv4));
    ipv4->packet_id = 0;
    ipv4->fragment_offset = 0;
    ipv4->time_to_live = 64;
    ipv4->next_proto_id = IPPROTO_UDP;
    ipv4->src_addr = src_ip;
    ipv4->dst_addr = dst_ip;

    ipv4->hdr_checksum = 0;
    ipv4->hdr_checksum = rte_ipv4_cksum(ipv4);

    return mbuf;
}

static int test_ip4_lookup_processing(void)
{
    const uint16_t num_packets = RTE_DIM(test_packets);
    struct rte_mbuf *mbufs[num_packets];
    void *obj_ptrs[num_packets];
    int failures = 0;

    for (int i = 0; i < num_packets; i++) {
        mbufs[i] = create_test_packet(
            test_packets[i].src_ip,
            test_packets[i].dst_ip
        );
        if (!mbufs[i]) {
            printf("Failed to create packet %d\n", i);
            failures++;
            continue;
        }
        obj_ptrs[i] = mbufs[i];
    }
    test_graph = rte_graph_lookup("ip4_lookup_test");
    if (test_graph == NULL) {
        printf("Error: Graph lookup failed.\n");
        return -1;
    }
    uint16_t processed = ip4_lookup_node->process(
        test_graph,
        ip4_lookup_node,
        obj_ptrs,
        num_packets
    );

    if (processed != num_packets) {
        printf("Error: Processed %u packets, expected %u\n",
               processed, num_packets);
        failures++;
    }

    for (int i = 0; i < num_packets; i++) {
        if (!mbufs[i])  // 忽略创建失败的包
            continue;

        uint64_t next_hop = GET_NEXT_HOP(mbufs[i]);
        uint8_t expected = test_packets[i].expected_next_hop;

        // if (next_node != test_packets[i].expected_next_hop) {
        //     printf("Packet %d: Next node mismatch! Got %u, expected %u\n",
        //            i, next_node, test_packets[i].expected_next_hop);
        //     failures++;
        // }
        
        if (expected == RTE_NODE_IP4_LOOKUP_NEXT_PKT_DROP) {
            // 如果预期是丢弃，则无需检查 next_hop
            continue;
        }

        // if (expected  != RTE_NODE_IP4_LOOKUP_NEXT_PKT_DROP) {
        //     if (next_hop != test_packets[i].expected_next_hop) {
        //         printf("Packet %d: Next hop mismatch! Got %lu, expected %u\n",
        //                i, next_hop, test_packets[i].expected_next_hop);
        //         failures++;
        //     }
        // }
        if (next_hop != expected) {
            printf("Packet %d: Next hop mismatch! Got %lu, expected %u\n",
                   i, next_hop, expected);
            failures++;
        }
    }
    for (int i = 0; i < num_packets; i++) {
        if (mbufs[i])
            rte_pktmbuf_free(mbufs[i]);
    }
    return failures ? -1 : 0;
}

static struct unit_test_suite ip4_lookup_testsuite = {
    .suite_name = "IP4 Lookup Node Unit Test",
    .setup = test_ip4_lookup_setup,
    .teardown = test_ip4_lookup_teardown,
    .unit_test_cases = {
        TEST_CASE(test_ip4_lookup_processing),
        TEST_CASES_END()
    }
};

static int test_ip4_lookup(void)
{
    return unit_test_suite_runner(&ip4_lookup_testsuite);
}

// REGISTER_TEST_COMMAND(ip4_lookup_node_autotest, test_ip4_lookup);
REGISTER_FAST_TEST(ip4_lookup_node_autotest, false, true, test_ip4_lookup);