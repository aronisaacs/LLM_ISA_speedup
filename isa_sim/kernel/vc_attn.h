// Decode attention over one KV head, for gem5.
//
// A self-contained copy of the attention loop in llama.cpp's CPU flash
// attention (ggml_compute_forward_flash_attn_ext_f16_one_chunk): for each query
// head, walk every cached token, score q.k, and fold v into an online softmax.
// Formats:
//   f16    dense keys and values, the baseline
//   vc50   vck50 keys and vcv50 values, byte-identical to llama.cpp's blocks
//
// Nothing here depends on ggml, so it cross-compiles to a static binary for
// gem5's syscall-emulation mode.

#pragma once

#include <stddef.h>
#include <stdint.h>

#define VC_DIM 128
#define VCK50_KEEP_PAIRS 32
#define VCV50_KEEP 64

typedef __fp16 vc_f16;

typedef struct {
    uint8_t mask[VC_DIM / 16];
    vc_f16  vals[2 * VCK50_KEEP_PAIRS];
} vc_block_k;

typedef struct {
    uint8_t mask[VC_DIM / 8];
    vc_f16  vals[VCV50_KEEP];
} vc_block_v;

_Static_assert(sizeof(vc_block_k) == 136, "vck50 block must match ggml");
_Static_assert(sizeof(vc_block_v) == 144, "vcv50 block must match ggml");

// Compression, as in llama.cpp's quantize_row_vck50_ref / quantize_row_vcv50_ref.
void vc_compress_k(const float * x, vc_block_k * y);
void vc_compress_v(const float * x, vc_block_v * y);
void vc_decompress_v(const vc_block_v * x, float * y);

// out[h] = softmax(scale * q[h] . K) V for heads h < n_q, over n_kv tokens.
// q and out are n_q x VC_DIM floats.
void attn_f16(const float * q, int n_q, const vc_f16 * k, const vc_f16 * v, int n_kv, float scale, float * out);
void attn_vc50(const float * q, int n_q, const vc_block_k * k, const vc_block_v * v, int n_kv, float scale, float * out);
