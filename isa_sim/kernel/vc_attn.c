#include "vc_attn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float score;
    int   index;
} vc_entry;

// Larger score first; on a tie the lower index first (same as llama.cpp).
static int vc_entry_cmp(const void * a, const void * b) {
    const vc_entry * x = (const vc_entry *) a;
    const vc_entry * y = (const vc_entry *) b;
    if (x->score != y->score) {
        return x->score > y->score ? -1 : 1;
    }
    return x->index - y->index;
}

static void vc_top_mask(const float * score, int n, int n_keep, uint8_t * mask) {
    vc_entry e[VC_DIM];
    for (int j = 0; j < n; ++j) {
        e[j].score = score[j];
        e[j].index = j;
    }
    qsort(e, n, sizeof(vc_entry), vc_entry_cmp);
    memset(mask, 0, (n + 7) / 8);
    for (int r = 0; r < n_keep; ++r) {
        mask[e[r].index / 8] |= (uint8_t) (1u << (e[r].index % 8));
    }
}

static inline int vc_bit(const uint8_t * mask, int j) {
    return (mask[j / 8] >> (j % 8)) & 1;
}

void vc_compress_k(const float * x, vc_block_k * y) {
    float score[VC_DIM / 2];
    for (int j = 0; j < VC_DIM / 2; ++j) {
        score[j] = x[2*j]*x[2*j] + x[2*j + 1]*x[2*j + 1];
    }
    vc_top_mask(score, VC_DIM / 2, VCK50_KEEP_PAIRS, y->mask);
    int o = 0;
    for (int j = 0; j < VC_DIM / 2; ++j) {
        if (vc_bit(y->mask, j)) {
            y->vals[o++] = (vc_f16) x[2*j];
            y->vals[o++] = (vc_f16) x[2*j + 1];
        }
    }
}

void vc_compress_v(const float * x, vc_block_v * y) {
    float score[VC_DIM];
    for (int j = 0; j < VC_DIM; ++j) {
        score[j] = fabsf(x[j]);
    }
    vc_top_mask(score, VC_DIM, VCV50_KEEP, y->mask);
    int o = 0;
    for (int j = 0; j < VC_DIM; ++j) {
        if (vc_bit(y->mask, j)) {
            y->vals[o++] = (vc_f16) x[j];
        }
    }
}

void vc_decompress_v(const vc_block_v * x, float * y) {
    int o = 0;
    for (int j = 0; j < VC_DIM; ++j) {
        y[j] = vc_bit(x->mask, j) ? (float) x->vals[o++] : 0.0f;
    }
}

// q.k over the kept pairs only.
static float dot_vck50(const vc_block_k * k, const vc_f16 * q) {
    float sum = 0.0f;
    int o = 0;
    for (int j = 0; j < VC_DIM / 2; ++j) {
        if (vc_bit(k->mask, j)) {
            sum += (float) k->vals[o]     * (float) q[2*j];
            sum += (float) k->vals[o + 1] * (float) q[2*j + 1];
            o += 2;
        }
    }
    return sum;
}

static float dot_f16(const vc_f16 * k, const vc_f16 * q) {
    float sum = 0.0f;
    for (int j = 0; j < VC_DIM; ++j) {
        sum += (float) k[j] * (float) q[j];
    }
    return sum;
}

// One step of the online softmax: fold value row v with score s into acc.
static inline void softmax_step(float s, const float * v, float * acc, float * M, float * S) {
    float ms = 1.0f;
    float vs = 1.0f;
    if (s > *M) {
        ms = expf(*M - s);
        *M = s;
        for (int d = 0; d < VC_DIM; ++d) {
            acc[d] *= ms;
        }
    } else {
        vs = expf(s - *M);
    }
    for (int d = 0; d < VC_DIM; ++d) {
        acc[d] += vs * v[d];
    }
    *S = *S * ms + vs;
}

static inline void finish(const float * acc, float S, float * out) {
    for (int d = 0; d < VC_DIM; ++d) {
        out[d] = acc[d] / S;
    }
}

// Like llama.cpp, dense f16 values are accumulated in f16 (ggml_vec_mad_f16 into
// VKQ16); compressed values are decompressed to f32 and accumulated in f32.
void attn_f16(const float * q, int n_q, const vc_f16 * k, const vc_f16 * v, int n_kv, float scale, float * out) {
    vc_f16 qh[VC_DIM];
    vc_f16 acc[VC_DIM];
    for (int h = 0; h < n_q; ++h) {
        for (int d = 0; d < VC_DIM; ++d) {
            qh[d] = (vc_f16) q[h*VC_DIM + d];
            acc[d] = 0;
        }
        float M = -INFINITY;
        float S = 0.0f;
        for (int t = 0; t < n_kv; ++t) {
            const float s = scale * dot_f16(k + (size_t) t*VC_DIM, qh);
            const vc_f16 * vrow = v + (size_t) t*VC_DIM;
            float ms = 1.0f;
            float vs = 1.0f;
            if (s > M) {
                ms = expf(M - s);
                M = s;
                const vc_f16 ms16 = (vc_f16) ms;
                for (int d = 0; d < VC_DIM; ++d) {
                    acc[d] = acc[d] * ms16;
                }
            } else {
                vs = expf(s - M);
            }
            const vc_f16 vs16 = (vc_f16) vs;
            for (int d = 0; d < VC_DIM; ++d) {
                acc[d] = acc[d] + vrow[d] * vs16;
            }
            S = S * ms + vs;
        }
        for (int d = 0; d < VC_DIM; ++d) {
            out[h*VC_DIM + d] = (float) acc[d] / S;
        }
    }
}

void attn_vc50(const float * q, int n_q, const vc_block_k * k, const vc_block_v * v, int n_kv, float scale, float * out) {
    vc_f16 qh[VC_DIM];
    float vrow[VC_DIM];
    float acc[VC_DIM];
    for (int h = 0; h < n_q; ++h) {
        for (int d = 0; d < VC_DIM; ++d) {
            qh[d] = (vc_f16) q[h*VC_DIM + d];
        }
        memset(acc, 0, sizeof(acc));
        float M = -INFINITY;
        float S = 0.0f;
        for (int t = 0; t < n_kv; ++t) {
            const float s = scale * dot_vck50(k + t, qh);
            vc_decompress_v(v + t, vrow);
            softmax_step(s, vrow, acc, &M, &S);
        }
        finish(acc, S, out + h*VC_DIM);
    }
}
