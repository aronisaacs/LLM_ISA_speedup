// Checks the standalone kernel against llama.cpp's ggml, on the Mac.
//
// 1. vc_compress_k / vc_compress_v produce the same bytes as ggml's vck50/vcv50.
// 2. attn_f16 and attn_vc50 match ggml_flash_attn_ext on the CPU backend with
//    f16 and vck50/vcv50 caches (the op llama.cpp runs for attention).

#include "vc_attn.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_KV 1024
#define N_Q  4   // query heads sharing the KV head (Llama 3.1 8B GQA)

static uint64_t rng = 0x9e3779b97f4a7c15ull;

static float randn(void) {
    // Box-Muller from a 64-bit LCG; enough for test data.
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    const double u1 = ((rng >> 11) + 1.0) / 9007199254740993.0;
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    const double u2 = (rng >> 11) / 9007199254740992.0;
    return (float) (sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

static void fill(float * x, int rows, const float * scale) {
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < VC_DIM; ++d) {
            x[r*VC_DIM + d] = (float) (vc_f16) (randn() * scale[d]);
        }
    }
}

// ggml_flash_attn_ext over one KV head; out is N_Q x VC_DIM.
static void ggml_attn(enum ggml_type tk, enum ggml_type tv, const float * q, const float * k, const float * v,
                      float scale, float * out) {
    struct ggml_init_params ip = { 256u*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, VC_DIM, 1, N_Q, 1);
    struct ggml_tensor * K = ggml_new_tensor_4d(ctx, tk, VC_DIM, N_KV, 1, 1);
    struct ggml_tensor * V = ggml_new_tensor_4d(ctx, tv, VC_DIM, N_KV, 1, 1);
    memcpy(Q->data, q, sizeof(float) * N_Q * VC_DIM);
    for (int t = 0; t < N_KV; ++t) {
        ggml_get_type_traits_cpu(tk)->from_float(k + t*VC_DIM, (char *) K->data + t*K->nb[1], VC_DIM);
        ggml_get_type_traits_cpu(tv)->from_float(v + t*VC_DIM, (char *) V->data + t*V->nb[1], VC_DIM);
    }

    struct ggml_tensor * O = ggml_flash_attn_ext(ctx, Q, K, V, NULL, scale, 0.0f, 0.0f);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, O);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    memcpy(out, O->data, sizeof(float) * N_Q * VC_DIM);  // [D, n_head, n_batch]
    ggml_free(ctx);
}

static double max_rel(const float * a, const float * b, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        num = fmax(num, fabs((double) a[i] - b[i]));
        den = fmax(den, fabs((double) b[i]));
    }
    return num / den;
}

int main(void) {
    ggml_cpu_init();

    float scale_k[VC_DIM], scale_v[VC_DIM], scale_q[VC_DIM];
    for (int d = 0; d < VC_DIM; ++d) {
        scale_k[d] = expf(randn());
        scale_v[d] = expf(0.5f * randn());
        scale_q[d] = 1.0f;
    }
    float * k = malloc(sizeof(float) * N_KV * VC_DIM);
    float * v = malloc(sizeof(float) * N_KV * VC_DIM);
    float q[N_Q * VC_DIM];
    fill(k, N_KV, scale_k);
    fill(v, N_KV, scale_v);
    fill(q, N_Q, scale_q);
    const float scale = 1.0f / sqrtf((float) VC_DIM);

    // 1. packing
    int bad_k = 0, bad_v = 0;
    for (int t = 0; t < N_KV; ++t) {
        vc_block_k bk; vc_block_v bv;
        char gk[sizeof(vc_block_k)], gv[sizeof(vc_block_v)];
        vc_compress_k(k + t*VC_DIM, &bk);
        vc_compress_v(v + t*VC_DIM, &bv);
        ggml_get_type_traits_cpu(GGML_TYPE_VCK50)->from_float(k + t*VC_DIM, gk, VC_DIM);
        ggml_get_type_traits_cpu(GGML_TYPE_VCV50)->from_float(v + t*VC_DIM, gv, VC_DIM);
        bad_k += memcmp(&bk, gk, sizeof bk) != 0;
        bad_v += memcmp(&bv, gv, sizeof bv) != 0;
    }
    printf("packing: %d/%d keys and %d/%d values differ from ggml\n", bad_k, N_KV, bad_v, N_KV);

    // 2. attention
    vc_f16 * k16 = malloc(sizeof(vc_f16) * N_KV * VC_DIM);
    vc_f16 * v16 = malloc(sizeof(vc_f16) * N_KV * VC_DIM);
    vc_block_k * kc = malloc(sizeof(vc_block_k) * N_KV);
    vc_block_v * vc = malloc(sizeof(vc_block_v) * N_KV);
    for (int i = 0; i < N_KV * VC_DIM; ++i) {
        k16[i] = (vc_f16) k[i];
        v16[i] = (vc_f16) v[i];
    }
    for (int t = 0; t < N_KV; ++t) {
        vc_compress_k(k + t*VC_DIM, kc + t);
        vc_compress_v(v + t*VC_DIM, vc + t);
    }

    float ours[N_Q * VC_DIM], theirs[N_Q * VC_DIM];
    attn_f16(q, N_Q, k16, v16, N_KV, scale, ours);
    ggml_attn(GGML_TYPE_F16, GGML_TYPE_F16, q, k, v, scale, theirs);
    const double e16 = max_rel(ours, theirs, N_Q * VC_DIM);
    printf("attn f16  vs ggml: max abs error / max |out| = %.2e\n", e16);

    attn_vc50(q, N_Q, kc, vc, N_KV, scale, ours);
    ggml_attn(GGML_TYPE_VCK50, GGML_TYPE_VCV50, q, k, v, scale, theirs);
    const double evc = max_rel(ours, theirs, N_Q * VC_DIM);
    printf("attn vc50 vs ggml: max abs error / max |out| = %.2e\n", evc);

    // f16 accumulates in f16 on both sides, but ggml splits long caches into chunks
    // and merges them (use_split_kv_path), so the rounding differs at the f16 ulp level.
    const int ok = bad_k == 0 && bad_v == 0 && e16 < 5e-3 && evc < 1e-4;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
