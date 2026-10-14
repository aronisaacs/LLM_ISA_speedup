// Runs llama.cpp's vector-compressed cache types on golden inputs.
//
//   vc_roundtrip <type> <in.f32> <query.f32> <out.f32> <dots.f32>
//
// <in.f32> holds n vectors of 128 floats in llama.cpp's layout. Each is
// compressed with the CPU backend's from_float (the path the KV cache write
// takes) and decompressed with to_float into <out.f32>. For vck50, each vector is
// also dotted with the matching query row through the type's vec_dot (the path
// flash attention takes) into <dots.f32>.

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 128

static float * read_floats(const char * path, long * count) {
    FILE * f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *count = ftell(f) / (long) sizeof(float);
    fseek(f, 0, SEEK_SET);
    float * data = malloc(*count * sizeof(float));
    if (fread(data, sizeof(float), *count, f) != (size_t) *count) { perror(path); exit(1); }
    fclose(f);
    return data;
}

static void write_floats(const char * path, const float * data, long count) {
    FILE * f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(data, sizeof(float), count, f);
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <vck50|vcv50> <in.f32> <query.f32> <out.f32> <dots.f32>\n", argv[0]);
        return 1;
    }
    ggml_cpu_init();

    enum ggml_type type;
    if (strcmp(argv[1], "vck50") == 0) {
        type = GGML_TYPE_VCK50;
    } else if (strcmp(argv[1], "vcv50") == 0) {
        type = GGML_TYPE_VCV50;
    } else {
        fprintf(stderr, "unknown type %s\n", argv[1]);
        return 1;
    }

    long count, qcount;
    float * in = read_floats(argv[2], &count);
    float * query = read_floats(argv[3], &qcount);
    if (count % DIM != 0 || qcount != count) {
        fprintf(stderr, "inputs must be the same number of %d-float vectors\n", DIM);
        return 1;
    }
    const long n = count / DIM;

    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    const struct ggml_type_traits_cpu * cpu = ggml_get_type_traits_cpu(type);
    const size_t row = ggml_row_size(type, DIM);

    char * packed = malloc(n * row);
    float * out = malloc(count * sizeof(float));
    float * dots = calloc(n, sizeof(float));
    ggml_fp16_t * q16 = malloc(DIM * sizeof(ggml_fp16_t));

    for (long i = 0; i < n; ++i) {
        cpu->from_float(in + i*DIM, packed + i*row, DIM);
        traits->to_float(packed + i*row, out + i*DIM, DIM);
        if (cpu->vec_dot) {
            ggml_get_type_traits_cpu(cpu->vec_dot_type)->from_float(query + i*DIM, q16, DIM);
            cpu->vec_dot(DIM, dots + i, 0, packed + i*row, 0, q16, 0, 1);
        }
    }

    write_floats(argv[4], out, count);
    write_floats(argv[5], dots, n);
    printf("%s: %ld vectors, %zu bytes each (dense f16: %d)\n", argv[1], n, row, 2*DIM);
    return 0;
}
