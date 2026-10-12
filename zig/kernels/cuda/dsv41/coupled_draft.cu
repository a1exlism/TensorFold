// Coupled drafts for the DSpark Markov loop at temperature > 0 (TF_DS_COUPLED_DRAFTS). A step's score is the
// drafter's logit plus the Markov bias of the previous token, as markov.py's _score; instead of the argmax, each row
// draws its token the way the target's sampler (sampling.zig sampleRow) will draw the token at that position: its
// top_k candidates (value, then lower id), the top_p and min_p cuts on their scaled probabilities, then the first best
// of scaled value + lambda * gumbel, the gumbel from the target's keyed uniform at that position. lambda 1: the very
// draw the target makes, on the drafter's scores; 0: the argmax (the original loop's pick); in between: a sharper
// drafter on the same noise. The target's tokens never change (verification keeps the target's own draw): a draft
// lands more often when the drafter and the target agree on the winner of the same noise. Temperature 0: the argmax.
#include <cuda_fp16.h>
#include <stdint.h>

#define KMAX 32
#define CD_CH 8
#define CD_WMAX 8
#define FULL 0xffffffffu

__device__ __forceinline__ bool before(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

// A warp's sorted list (lane i holds the i-th best under `before`): insert the lanes' items (s, g) of mask m that still
// beat the list's K-th item (tv, ti), keeping that K-th item current.
__device__ __forceinline__ void warp_insert(float& lv, int& li, float s, int g, unsigned m, int K, float& tv, int& ti) {
    const int lane = threadIdx.x & 31;
    while (m) {
        const int src = __ffs(m) - 1;
        m &= m - 1;
        const float nv = __shfl_sync(FULL, s, src);
        const int ni = __shfl_sync(FULL, g, src);
        const float pv = __shfl_up_sync(FULL, lv, 1);
        const int pi = __shfl_up_sync(FULL, li, 1);
        if (before(nv, ni, lv, li)) {
            if (lane > 0 && before(nv, ni, pv, pi)) {
                lv = pv;
                li = pi;
            } else {
                lv = nv;
                li = ni;
            }
        }
        tv = __shfl_sync(FULL, lv, K - 1);
        ti = __shfl_sync(FULL, li, K - 1);
        m &= __ballot_sync(FULL, before(s, g, tv, ti));
    }
}

// Program r (1024 threads): row r's scores over this rank's columns, its K best (value desc, id asc) into
// SEND[r * 2K]: the values in [0, K), the ids (as floats, exact below 2^24) in [K, 2K). Each warp keeps a sorted
// list of its columns' best (8 chunks of 32 columns offered while the next 8 load), then the lists merge pairwise.
extern "C" __global__ void __launch_bounds__(1024) dsv41_coupled_topk(const float* __restrict__ LG, long long l_row,
                                                                      const long long* __restrict__ TOK, int t_stride,
                                                                      const int* __restrict__ SLOT,
                                                                      const __half* __restrict__ CACHE, long long c_row,
                                                                      const __half* __restrict__ STAGE, long long s_row,
                                                                      int n_cols, int seg0, float* __restrict__ SEND,
                                                                      int K) {
    const int r = blockIdx.x;
    const long long tok = TOK[(long long)r * t_stride];
    const int slot = SLOT[tok];
    const __half* brow = slot >= 0 ? CACHE + (long long)slot * c_row : STAGE + (long long)r * s_row;
    const float* lrow = LG + (long long)r * l_row;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (K > KMAX) K = KMAX;
    if (K < 1) K = 1;
    float lv = -INFINITY, tv = -INFINITY;
    int li = 0x7fffffff, ti = 0x7fffffff;
    const int gofs = seg0 * n_cols;
    const int stride = nw * 32 * CD_CH;
    float a[CD_CH], b[CD_CH];
    int base = warp * 32 * CD_CH;
#pragma unroll
    for (int u = 0; u < CD_CH; ++u) {
        const int c = base + u * 32 + lane;
        a[u] = c < n_cols ? lrow[c] : -INFINITY;
        b[u] = c < n_cols ? __half2float(brow[c]) : 0.f;
    }
    for (; base < n_cols; base += stride) {
        float s[CD_CH];
#pragma unroll
        for (int u = 0; u < CD_CH; ++u) s[u] = a[u] + b[u];
        const int nb = base + stride;
#pragma unroll
        for (int u = 0; u < CD_CH; ++u) {
            const int c = nb + u * 32 + lane;
            a[u] = c < n_cols ? lrow[c] : -INFINITY;
            b[u] = c < n_cols ? __half2float(brow[c]) : 0.f;
        }
#pragma unroll
        for (int u = 0; u < CD_CH; ++u) {
            const int c = base + u * 32 + lane;
            const int g = c < n_cols ? gofs + c : 0x7fffffff;
            const float v = c < n_cols ? s[u] : -INFINITY;
            const unsigned m = __ballot_sync(FULL, before(v, g, tv, ti));
            if (m) warp_insert(lv, li, v, g, m, K, tv, ti);
        }
    }
    __shared__ float sv[32 * 32];
    __shared__ int si[32 * 32];
    for (int step = 1; step < nw; step <<= 1) {
        if ((warp & (2 * step - 1)) == step) {
            sv[warp * 32 + lane] = lv;
            si[warp * 32 + lane] = li;
        }
        __syncthreads();
        if ((warp & (2 * step - 1)) == 0 && warp + step < nw) {
            const float s = sv[(warp + step) * 32 + lane];
            const int g = si[(warp + step) * 32 + lane];
            const unsigned m = __ballot_sync(FULL, before(s, g, tv, ti));
            if (m) warp_insert(lv, li, s, g, m, K, tv, ti);
        }
        __syncthreads();
    }
    if (warp == 0 && lane < K) {
        SEND[(long long)r * 2 * K + lane] = lv;
        SEND[(long long)r * 2 * K + K + lane] = (float)li;
    }
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// exact_sampling.uniform's bits: a (0, 1) value from a splitmix64 chain of (seed, position, token id), as the integer
// numerator over 2^53 (u = (b + 0.5) / 2^53)
__device__ __forceinline__ unsigned long long keyed_bits(unsigned long long seed, unsigned long long pos, unsigned long long id) {
    unsigned long long x = mix64(seed + 0x9E3779B97F4A7C15ull);
    x = mix64(x ^ (pos * 0xD1B54A32D192ED03ull));
    x = mix64(x ^ id);
    return x >> 11;
}

// -log(-log(u)) in float for u = (b + 0.5) / 2^53, accurate at both ends: -log(u) from u when u < 1/2, else from 1 - u
__device__ __forceinline__ float gumbel_of(unsigned long long b) {
    const unsigned long long half = 1ull << 52;
    float e1;
    if (b < half) {
        e1 = -logf(((float)b + 0.5f) * 0x1p-53f);
    } else {
        const unsigned long long rest = (1ull << 53) - b;  // 1 - u = (rest - 0.5) / 2^53
        e1 = -log1pf(-(((float)rest - 0.5f) * 0x1p-53f));
    }
    return -logf(e1);
}

// One warp: each row's candidates from every rank's K best (G [world, rows, 2K], each rank's list in order) merged
// under (value desc, id asc) by counting, then the drawn token (sampleRow's rule at position Q0[r] + 1 + step, the
// noise scaled by lam; lane j the j-th candidate) into OUT[r * o_stride].
extern "C" __global__ void dsv41_coupled_pick(const float* __restrict__ G, int rows, int world, int K,
                                              long long* __restrict__ OUT, int o_stride,
                                              const unsigned long long* __restrict__ SEED,
                                              const long long* __restrict__ Q0, int step, const float* __restrict__ TEMP,
                                              const int* __restrict__ TOPK, const float* __restrict__ TOPP,
                                              const float* __restrict__ MINP, float lam) {
    __shared__ float gv[CD_WMAX * KMAX];
    __shared__ int gi[CD_WMAX * KMAX];
    __shared__ float mv[CD_WMAX * KMAX];
    __shared__ int mi[CD_WMAX * KMAX];
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    if (world > CD_WMAX) world = CD_WMAX;
    if (K > KMAX) K = KMAX;
    const int nt = world * K;
    for (int r = 0; r < rows; ++r) {
        for (int t = lane; t < nt; t += 32) {
            const int w = t / K, j = t - w * K;
            const float* g = G + ((long long)w * rows + r) * 2 * K;
            gv[t] = g[j];
            gi[t] = (int)g[K + j];
        }
        __syncwarp();
        int valid = 0;
        for (int t = lane; t < nt; t += 32) {
            const int w = t / K, j = t - w * K;
            const float v = gv[t];
            const int id = gi[t];
            int pos = j;
            for (int w2 = 0; w2 < world; ++w2) {
                if (w2 == w) continue;
                int lo = 0, hi = K;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (before(gv[w2 * K + mid], gi[w2 * K + mid], v, id)) lo = mid + 1;
                    else hi = mid;
                }
                pos += lo;
            }
            mv[pos] = v;
            mi[pos] = id;
            valid += v > -INFINITY ? 1 : 0;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) valid += __shfl_xor_sync(FULL, valid, o);
        __syncwarp();
        const float temp = TEMP[r];
        int pick = 0;
        if (valid > 0) {
            pick = mi[0];
            if (temp > 0.f && lam > 0.f) {
                int k = TOPK[r];
                if (k > valid) k = valid;
                if (k > 32) k = 32;
                if (k < 1) k = 1;
                const float inv_t = 1.0f / fmaxf(temp, 1e-6f);
                const bool on = lane < k;
                const float sc = on ? mv[lane] * inv_t : -INFINITY;
                float score = on ? sc + lam * gumbel_of(keyed_bits(SEED[r], (unsigned long long)(Q0[r] + 1 + step),
                                                                   (unsigned long long)(long long)mi[lane]))
                                 : -INFINITY;
                const float top = __shfl_sync(FULL, sc, 0);
                const float top_p = TOPP[r];
                if (0.f < top_p && top_p < 1.f) {
                    const float p = on ? expf(sc - top) : 0.f;
                    float sum = p;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(FULL, sum, o);
                    float acc = p / sum;
#pragma unroll
                    for (int o = 1; o < 32; o <<= 1) {
                        const float x = __shfl_up_sync(FULL, acc, o);
                        if (lane >= o) acc += x;
                    }
                    const int below = __popc(__ballot_sync(FULL, on && acc < top_p));
                    if (lane > below) score = -INFINITY;
                }
                const float min_p = MINP[r];
                if (min_p > 0.f && on && sc < top + logf(min_p)) score = -INFINITY;
                float bv = score;
                int bj = lane;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = __shfl_xor_sync(FULL, bv, o);
                    const int oj = __shfl_xor_sync(FULL, bj, o);
                    if (ov > bv || (ov == bv && oj < bj)) {
                        bv = ov;
                        bj = oj;
                    }
                }
                pick = mi[bj];
            }
        }
        if (lane == 0) OUT[(long long)r * o_stride] = pick;
        __syncwarp();
    }
}
