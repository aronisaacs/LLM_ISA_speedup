// One decode step of attention over one KV head, for gem5.
//
//   bench <f16|vc50> <n_kv>
//
// Builds a synthetic cache of n_kv tokens, compresses it outside the measured
// region, then runs attention for the 4 query heads that share the KV head.
// Built with -DM5OPS, the measured region is bracketed by m5 reset/dump stats,
// so gem5's stats.txt covers only the attention.

#include "vc_attn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef M5OPS
#include <gem5/m5ops.h>
#define ROI_BEGIN() m5_reset_stats(0, 0)
#define ROI_END()   m5_dump_stats(0, 0)
#else
#define ROI_BEGIN()
#define ROI_END()
#endif

#define N_Q 4

static uint32_t rng = 12345;

static float uniform(void) {
    rng = rng * 1664525u + 1013904223u;
    return (float) (rng >> 8) / 16777216.0f * 2.0f - 1.0f;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <f16|vc50> <n_kv>\n", argv[0]);
        return 1;
    }
    const int vc = strcmp(argv[1], "vc50") == 0;
    if (!vc && strcmp(argv[1], "f16") != 0) {
        fprintf(stderr, "unknown format %s\n", argv[1]);
        return 1;
    }
    const int n_kv = atoi(argv[2]);

    float row[VC_DIM];
    float q[N_Q * VC_DIM];
    float out[N_Q * VC_DIM];
    for (int i = 0; i < N_Q * VC_DIM; ++i) {
        q[i] = uniform();
    }

    vc_f16 * k16 = NULL, * v16 = NULL;
    vc_block_k * kc = NULL;
    vc_block_v * vcb = NULL;
    if (vc) {
        kc = malloc(sizeof(vc_block_k) * n_kv);
        vcb = malloc(sizeof(vc_block_v) * n_kv);
    } else {
        k16 = malloc(sizeof(vc_f16) * VC_DIM * n_kv);
        v16 = malloc(sizeof(vc_f16) * VC_DIM * n_kv);
    }
    for (int t = 0; t < n_kv; ++t) {
        for (int d = 0; d < VC_DIM; ++d) row[d] = uniform();
        if (vc) vc_compress_k(row, kc + t); else for (int d = 0; d < VC_DIM; ++d) k16[(size_t) t*VC_DIM + d] = (vc_f16) row[d];
        for (int d = 0; d < VC_DIM; ++d) row[d] = uniform();
        if (vc) vc_compress_v(row, vcb + t); else for (int d = 0; d < VC_DIM; ++d) v16[(size_t) t*VC_DIM + d] = (vc_f16) row[d];
    }

    const float scale = 1.0f / sqrtf((float) VC_DIM);
    ROI_BEGIN();
    if (vc) {
        attn_vc50(q, N_Q, kc, vcb, n_kv, scale, out);
    } else {
        attn_f16(q, N_Q, k16, v16, n_kv, scale, out);
    }
    ROI_END();

    double checksum = 0.0;
    for (int i = 0; i < N_Q * VC_DIM; ++i) {
        checksum += out[i];
    }
    const size_t bytes = vc ? (sizeof(vc_block_k) + sizeof(vc_block_v)) * (size_t) n_kv
                            : 2 * sizeof(vc_f16) * VC_DIM * (size_t) n_kv;
    printf("%s n_kv=%d cache_bytes=%zu checksum=%.6f\n", argv[1], n_kv, bytes, checksum);
    return 0;
}
