// The 16x16x16 bf16 tile product both vendors' matrix cores do, one wave (AMD) or warp (NVIDIA) of 32 lanes:
// C[r][n] += sum_k A[r][k] * B[n][k], fp32 C, both operands stored row by row along k. Each lane holds its own
// share of the operands (Frag) and eight C entries, entry t at (acc_row(lane, t), acc_col(lane, t)).
//
// AMD: gfx12 WMMA, lane l holds row l % 16 at k (l / 16) * 8 .. +7 and C column l % 16, rows (l / 16) * 8 + t.
// NVIDIA: two mma.sync m16n8k16 (PTX ISA, "Matrix Fragments for mma.m16n8k16 with floating point type"); with
// g = l / 4 and c = 2 (l % 4) a lane holds (g, c), (g + 8, c), (g, c + 8), (g + 8, c + 8), two k each.
// MMA_NV_LAYOUT builds the NVIDIA layout under HIP with the product done by shuffles, so it can be checked on AMD.
#pragma once
#include "gpu.h"

#if defined(__CUDACC__) || defined(MMA_NV_LAYOUT)
#define MMA_NV 1
#endif

namespace mma_detail {
__device__ __forceinline__ float lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }
}  // namespace mma_detail

#ifdef MMA_NV

struct Frag { unsigned x, y, z, w; };
struct Acc {
    float v[8];
    __device__ float& operator[](int i) { return v[i]; }
    __device__ float operator[](int i) const { return v[i]; }
};

__device__ __forceinline__ int acc_row(int lane, int t) { return (lane >> 2) + ((t >> 1) & 1) * 8; }
__device__ __forceinline__ int acc_col(int lane, int t) { return (t >> 2) * 8 + (lane & 3) * 2 + (t & 1); }

// rows(r) points at row r's first k (4-byte aligned), or is null for a row of zeros.
template <class Rows>
__device__ __forceinline__ Frag frag_rows(int lane, Rows rows) {
    const int g = lane >> 2, c = (lane & 3) * 2;
    const bf16 *p0 = rows(g), *p1 = rows(g + 8);
    auto ld = [](const bf16* p, int k) { return p ? *(const unsigned*)(p + k) : 0u; };
    return Frag{ld(p0, c), ld(p1, c), ld(p0, c + 8), ld(p1, c + 8)};
}

// The tile at p (shared memory, rows 16-byte aligned, ld elements apart).
__device__ __forceinline__ Frag frag_lds(const bf16* p, int ld, int lane) {
#ifdef __CUDACC__
    // Lanes 0-7, 8-15, 16-23, 24-31 address rows 0-7 and 8-15 at k 0, then the same at k 8: Frag order.
    const unsigned a = (unsigned)__cvta_generic_to_shared(p + (lane & 15) * ld + (lane >> 4) * 8);
    Frag f;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(f.x), "=r"(f.y), "=r"(f.z), "=r"(f.w) : "r"(a));
    return f;
#else
    return frag_rows(lane, [&](int r) { return p + r * ld; });
#endif
}

// Entry by entry: at(r, k).
template <class At>
__device__ __forceinline__ Frag frag_at(int lane, At at) {
    const int g = lane >> 2, c = (lane & 3) * 2;
    auto pack = [&](int r, int k) { return (unsigned)bf16_bits(at(r, k)) | ((unsigned)bf16_bits(at(r, k + 1)) << 16); };
    return Frag{pack(g, c), pack(g + 8, c), pack(g, c + 8), pack(g + 8, c + 8)};
}

// D = A B + D for one m16n8k16 half; b0, b1 are B column n's k pairs (k 2j.., 2j + 8..) of lane n * 4 + j.
__device__ __forceinline__ void mma16816(float* d, const Frag& a, unsigned b0, unsigned b1) {
#ifdef __CUDACC__
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
#else
    using namespace mma_detail;
    auto dot2 = [](unsigned x, unsigned y, float acc) { return fmaf(hi(x), hi(y), fmaf(lo(x), lo(y), acc)); };
    const int lane = threadIdx.x & 31, g = lane >> 2, n0 = (lane & 3) * 2;
    for (int j = 0; j < 4; j++) {
        const int sa = g * 4 + j, s0 = n0 * 4 + j, s1 = s0 + 4;
        const unsigned a0 = __shfl(a.x, sa, 32), a1 = __shfl(a.y, sa, 32), a2 = __shfl(a.z, sa, 32), a3 = __shfl(a.w, sa, 32);
        const unsigned b00 = __shfl(b0, s0, 32), b01 = __shfl(b1, s0, 32), b10 = __shfl(b0, s1, 32), b11 = __shfl(b1, s1, 32);
        d[0] = dot2(a2, b01, dot2(a0, b00, d[0]));
        d[1] = dot2(a2, b11, dot2(a0, b10, d[1]));
        d[2] = dot2(a3, b01, dot2(a1, b00, d[2]));
        d[3] = dot2(a3, b11, dot2(a1, b10, d[3]));
    }
#endif
}

// B's rows g are its columns 0-7 of this half, rows g + 8 the other half.
__device__ __forceinline__ Acc mma(const Frag& a, const Frag& b, Acc c) {
    mma16816(c.v, a, b.x, b.z);
    mma16816(c.v + 4, a, b.y, b.w);
    return c;
}

// Each entry replaced by the max (sum) over its row of the tile: four entries per lane, across four lanes.
__device__ __forceinline__ Acc row_max(Acc v) {
    for (int h = 0; h < 2; h++) {
        float m = fmaxf(fmaxf(v[2 * h], v[2 * h + 1]), fmaxf(v[4 + 2 * h], v[5 + 2 * h]));
        m = fmaxf(m, __shfl_xor(m, 1, 32));
        m = fmaxf(m, __shfl_xor(m, 2, 32));
        v[2 * h] = v[2 * h + 1] = v[4 + 2 * h] = v[5 + 2 * h] = m;
    }
    return v;
}
__device__ __forceinline__ Acc row_sum(Acc v) {
    for (int h = 0; h < 2; h++) {
        float s = (v[2 * h] + v[2 * h + 1]) + (v[4 + 2 * h] + v[5 + 2 * h]);
        s += __shfl_xor(s, 1, 32);
        s += __shfl_xor(s, 2, 32);
        v[2 * h] = v[2 * h + 1] = v[4 + 2 * h] = v[5 + 2 * h] = s;
    }
    return v;
}

#else

typedef __bf16 Frag __attribute__((ext_vector_type(8)));
typedef float Acc __attribute__((ext_vector_type(8)));

__device__ __forceinline__ int acc_row(int lane, int t) { return (lane >> 4) * 8 + t; }
__device__ __forceinline__ int acc_col(int lane, int) { return lane & 15; }

template <class Rows>
__device__ __forceinline__ Frag frag_rows(int lane, Rows rows) {
    const bf16* p = rows(lane & 15);
    return __builtin_bit_cast(Frag, p ? *(const uint4*)(p + (lane >> 4) * 8) : uint4{0, 0, 0, 0});
}

__device__ __forceinline__ Frag frag_lds(const bf16* p, int ld, int lane) {
    return __builtin_bit_cast(Frag, *(const uint4*)(p + (lane & 15) * ld + (lane >> 4) * 8));
}

template <class At>
__device__ __forceinline__ Frag frag_at(int lane, At at) {
    Frag f;
    for (int e = 0; e < 8; e++) f[e] = at(lane & 15, (lane >> 4) * 8 + e);
    return f;
}

__device__ __forceinline__ Acc mma(Frag a, Frag b, Acc c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c); }

// A row's 16 columns sit in the 16 lanes that share l / 16.
__device__ __forceinline__ Acc row_max(Acc v) {
    for (int t = 0; t < 8; t++)
        for (int x = 1; x < 16; x <<= 1) v[t] = fmaxf(v[t], __shfl_xor(v[t], x, 32));
    return v;
}
__device__ __forceinline__ Acc row_sum(Acc v) {
    for (int t = 0; t < 8; t++)
        for (int x = 1; x < 16; x <<= 1) v[t] += __shfl_xor(v[t], x, 32);
    return v;
}

#endif
