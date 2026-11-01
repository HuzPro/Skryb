/* Greedy TDT decoding for Parakeet's prediction net and joint on the CPU (tdt_cpu.dll / libtdt_cpu.so).

   One call decodes a whole clip: the walk over frames (NeMo's GreedyTDTInfer, as parakeet_numpy's
   greedy_tdt_walk), the two-layer LSTM for each emitted token, and the joint over a window of frames for
   each prediction. Weights are int8 with one fp32 scale per output row, as openvino_int8_weights makes
   them; each product's input is rounded to int8 with one scale per vector just before it, and the products
   run as int8 dot products: AVX-VNNI-INT8 where the CPU has it, AVX2 otherwise, plain C as the last resort.

   Decoding reads the weights again for every token: 7 MB for the LSTM and 5.2 MB for each joint window,
   about 800 MB for 10 s of speech. One core can't pull that faster than about 16 GB/s, so the rows of each
   product are split among a few threads that spin between products and sleep between clips. Each thread
   keeps the same share every time, which stays in its core's cache; on a Core 5 320 two threads are
   fastest, as more land on the slower efficiency cores and hold the rest up. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#include <malloc.h>
#include <windows.h>
#pragma comment(lib, "Synchronization.lib")
#define EXPORT __declspec(dllexport)
#define TARGET_AVX2
#define TARGET_VNNI_INT8
#else
#include <cpuid.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#define EXPORT __attribute__((visibility("default")))
#define TARGET_AVX2 __attribute__((target("avx2")))
#define TARGET_VNNI_INT8 __attribute__((target("avx2,avxvnniint8")))
#endif
#include <immintrin.h>

/* The log-mel features in front of the encoder, as parakeet_numpy.log_mel_features makes them. */
#define N_FFT 512
#define HALF 256 /* the complex FFT's size: two real samples make one complex one */
#define HOP 160
#define BINS (N_FFT / 2 + 1)
#define MELS 128
#define LANES 8 /* frames transformed together, one in each AVX2 lane */
#define PREEMPHASIS 0.97f
#define LOG_GUARD 5.9604644775390625e-08f /* 2^-24 */

#define HIDDEN 640
#define GATES (4 * HIDDEN)
#define TOKENS 8193 /* the vocabulary and blank */
#define DURATIONS 5
#define OUTPUTS (TOKENS + DURATIONS)
#define BLANK 8192
#define MAX_SYMBOLS_PER_FRAME 10
#define WINDOW 16
#define FIRST_WINDOW 1
#define MAX_THREADS 16
/* About 50 us of spinning before a worker sleeps: products within a clip follow each other far sooner. */
#define SPINS 20000

enum { KERNEL_AUTO = 0, KERNEL_SCALAR = 1, KERNEL_AVX2 = 2, KERNEL_VNNI_INT8 = 3 };
static const int kDurations[DURATIONS] = {0, 1, 2, 3, 4};

/* Eight rows' dot products at once: rows 0-3 against x, rows 4-7 against y (the same vector for a plain
   product). Eight accumulators stay in registers and are summed together at the end, instead of one call and
   one horizontal sum per row. n is a multiple of 32. */
typedef void (*Dot8)(const int8_t* const rows[8], const int8_t* x, const int8_t* y, int n, int32_t out[8]);

typedef struct {
    int rows, cols;
    int8_t* w; /* rows x cols, 32-byte aligned */
    float* scale;
    float* bias;
} Linear;

typedef struct Tdt Tdt;
typedef void (*Task)(const Tdt* m, int thread, int begin, int end);

/* A product's rows, split evenly among the threads. */
typedef struct {
    Task task;
    volatile long rows;
    volatile long running; /* workers still busy with this product */
    volatile long generation;
    volatile long stop;
} Pool;

typedef struct {
    float h[2][HIDDEN], c[2][HIDDEN];
} State;

/* The window and the mel filters (nonzero weights only), and the FFT's twiddles and bit reversal. */
typedef struct {
    float window[N_FFT];
    int lo[MELS], count[MELS], offset[MELS];
    float* weights;
    float tw_re[HALF / 2], tw_im[HALF / 2];  /* e^(-2 pi i m / 256) */
    float post_re[BINS], post_im[BINS];      /* e^(-2 pi i k / 512) */
    int reverse[HALF];
} Mel;

/* What the products read and write, set before each one starts. */
typedef struct {
    const Linear* layer;
    const int8_t *xq, *hq;
    const float* c;
    float *h_out, *c_out;
    float sx, sh;
    const int8_t* pq;
    float ps;
    float* pred;
    int frames;
    int8_t q[WINDOW][HIDDEN];
    float s[WINDOW];
    float best_token[MAX_THREADS][WINDOW], best_duration[MAX_THREADS][WINDOW];
    int token[MAX_THREADS][WINDOW], duration[MAX_THREADS][WINDOW];
    const float* padded;
    float *log_mel, *features;
    int mel_frames;
} Work;

struct Tdt {
    Dot8 dot8;
    int kernel, threads;
    float* embed; /* TOKENS x HIDDEN */
    Linear lstm[2], pred, out;
    Pool pool;
    Work* work;
    Mel* mel;
#if defined(_MSC_VER)
    HANDLE workers[MAX_THREADS];
#else
    pthread_t workers[MAX_THREADS];
#endif
};

/* ---- atomics, sleeping and threads ------------------------------------------------------------------- */

#if defined(_MSC_VER)
static long add(volatile long* p, long v) { return _InterlockedExchangeAdd(p, v) + v; }
static long load(volatile long* p) { return _InterlockedCompareExchange(p, 0, 0); }
static void sleep_while(volatile long* p, long seen) { WaitOnAddress((volatile void*)p, &seen, sizeof(long), INFINITE); }
static void wake_all(volatile long* p) { WakeByAddressAll((void*)p); }
#else
static long add(volatile long* p, long v) { return __atomic_add_fetch(p, v, __ATOMIC_SEQ_CST); }
static long load(volatile long* p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
static void sleep_while(volatile long* p, long seen) {
    syscall(SYS_futex, (void*)p, FUTEX_WAIT_PRIVATE, (int)seen, NULL, NULL, 0);
}
static void wake_all(volatile long* p) { syscall(SYS_futex, (void*)p, FUTEX_WAKE_PRIVATE, 0x7fffffff, NULL, NULL, 0); }
#endif

static void take_rows(const Tdt* m, Pool* pool, int thread, int rows) {
    const int begin = (int)((long long)rows * thread / m->threads);
    const int end = (int)((long long)rows * (thread + 1) / m->threads);
    if (begin < end) pool->task(m, thread, begin, end);
}

typedef struct {
    Tdt* m;
    int thread;
} WorkerStart;

#if defined(_MSC_VER)
static DWORD WINAPI worker(void* arg)
#else
static void* worker(void* arg)
#endif
{
    WorkerStart* start = (WorkerStart*)arg;
    Tdt* m = start->m;
    const int thread = start->thread;
    free(start);
    Pool* pool = &m->pool;
    /* Nothing runs before tdt_create returns, so the first product is generation 1, even when this thread
       only starts after it was handed out. */
    long seen = 0;
    for (;;) {
        long now;
        int spins = 0;
        while ((now = load(&pool->generation)) == seen) {
            if (++spins < SPINS) {
                _mm_pause();
            } else {
                sleep_while(&pool->generation, seen);
                spins = 0;
            }
        }
        seen = now;
        if (load(&pool->stop)) break;
        take_rows(m, pool, thread, (int)pool->rows);
        add(&pool->running, -1);
    }
#if defined(_MSC_VER)
    return 0;
#else
    return NULL;
#endif
}

/* Run *task* over *rows* rows on every thread, the caller's included, and wait for all of them. */
static void run(Tdt* m, Task task, int rows) {
    Pool* pool = &m->pool;
    if (m->threads == 1) {
        task(m, 0, 0, rows);
        return;
    }
    pool->task = task;
    pool->rows = rows;
    pool->running = m->threads - 1;
    add(&pool->generation, 1);
    wake_all(&pool->generation);
    take_rows(m, pool, 0, rows);
    while (load(&pool->running) != 0) _mm_pause();
}

/* ---- dot products ---------------------------------------------------------------------------------- */

static void dot8_scalar(const int8_t* const r[8], const int8_t* x, const int8_t* y, int n, int32_t out[8]) {
    for (int k = 0; k < 8; k++) {
        const int8_t* v = k < 4 ? x : y;
        int32_t sum = 0;
        for (int i = 0; i < n; i++) sum += (int32_t)r[k][i] * (int32_t)v[i];
        out[k] = sum;
    }
}

/* Eight accumulators' lanes summed into one vector of eight totals, in row order. */
TARGET_AVX2 static inline __m256i sum8(__m256i a0, __m256i a1, __m256i a2, __m256i a3, __m256i a4, __m256i a5,
                                       __m256i a6, __m256i a7) {
    const __m256i s0123 = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    const __m256i s4567 = _mm256_hadd_epi32(_mm256_hadd_epi32(a4, a5), _mm256_hadd_epi32(a6, a7));
    return _mm256_add_epi32(_mm256_permute2x128_si256(s0123, s4567, 0x20), _mm256_permute2x128_si256(s0123, s4567, 0x31));
}

#define LOAD16(p) _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(p)))
#define ROW_AVX2(k, v) a##k = _mm256_add_epi32(a##k, _mm256_madd_epi16(LOAD16(r[k] + i), v))

TARGET_AVX2 static void dot8_avx2(const int8_t* const r[8], const int8_t* x, const int8_t* y, int n, int32_t out[8]) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (int i = 0; i < n; i += 16) {
        const __m256i vx = LOAD16(x + i), vy = LOAD16(y + i);
        ROW_AVX2(0, vx), ROW_AVX2(1, vx), ROW_AVX2(2, vx), ROW_AVX2(3, vx);
        ROW_AVX2(4, vy), ROW_AVX2(5, vy), ROW_AVX2(6, vy), ROW_AVX2(7, vy);
    }
    _mm256_storeu_si256((__m256i*)out, sum8(a0, a1, a2, a3, a4, a5, a6, a7));
}

#define LOAD32(p) _mm256_loadu_si256((const __m256i*)(p))
#define ROW_VNNI(k, v) a##k = _mm256_dpbssd_epi32(a##k, LOAD32(r[k] + i), v)

TARGET_VNNI_INT8 static void dot8_vnni_int8(const int8_t* const r[8], const int8_t* x, const int8_t* y, int n,
                                            int32_t out[8]) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (int i = 0; i < n; i += 32) {
        const __m256i vx = LOAD32(x + i), vy = LOAD32(y + i);
        ROW_VNNI(0, vx), ROW_VNNI(1, vx), ROW_VNNI(2, vx), ROW_VNNI(3, vx);
        ROW_VNNI(4, vy), ROW_VNNI(5, vy), ROW_VNNI(6, vy), ROW_VNNI(7, vy);
    }
    _mm256_storeu_si256((__m256i*)out, sum8(a0, a1, a2, a3, a4, a5, a6, a7));
}

static void cpuid(int leaf, int sub, int r[4]) {
#if defined(_MSC_VER)
    __cpuidex(r, leaf, sub);
#else
    unsigned a, b, c, d;
    __cpuid_count(leaf, sub, a, b, c, d);
    r[0] = (int)a, r[1] = (int)b, r[2] = (int)c, r[3] = (int)d;
#endif
}

/* The operating system must save the AVX registers too (XCR0 bits 1 and 2), not just the CPU have them. */
static int os_saves_avx(void) {
    int r[4];
    cpuid(1, 0, r);
    if (!((r[2] >> 27) & 1)) return 0;
#if defined(_MSC_VER)
    unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned eax, edx;
    __asm__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    unsigned long long xcr0 = ((unsigned long long)edx << 32) | eax;
#endif
    return (xcr0 & 6) == 6;
}

static int best_kernel(void) {
    int r[4];
    cpuid(0, 0, r);
    if (r[0] < 7 || !os_saves_avx()) return KERNEL_SCALAR;
    cpuid(7, 0, r);
    if (!((r[1] >> 5) & 1)) return KERNEL_SCALAR;
    cpuid(7, 1, r);
    return ((r[3] >> 4) & 1) ? KERNEL_VNNI_INT8 : KERNEL_AVX2;
}

/* ---- products ---------------------------------------------------------------------------------------- */

static void* aligned(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, 64);
#else
    void* p = NULL;
    return posix_memalign(&p, 64, bytes) == 0 ? p : NULL;
#endif
}

static void release(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

/* Rows are padded with zeros to a whole number of 8-row blocks. */
static int copy_linear(Linear* l, int rows, int cols, const int8_t* w, const float* scale, const float* bias) {
    const int padded = (rows + 7) / 8 * 8;
    l->rows = rows, l->cols = cols;
    l->w = (int8_t*)aligned((size_t)padded * cols);
    l->scale = (float*)aligned(sizeof(float) * padded);
    l->bias = (float*)aligned(sizeof(float) * padded);
    if (!l->w || !l->scale || !l->bias) return 0;
    memset(l->w, 0, (size_t)padded * cols);
    memset(l->scale, 0, sizeof(float) * padded);
    memset(l->bias, 0, sizeof(float) * padded);
    memcpy(l->w, w, (size_t)rows * cols);
    memcpy(l->scale, scale, sizeof(float) * rows);
    memcpy(l->bias, bias, sizeof(float) * rows);
    return 1;
}

static void free_linear(Linear* l) {
    release(l->w), release(l->scale), release(l->bias);
}

/* x as int8 with one scale: x ~ q * scale, rounding half to even. A zero vector has scale 0. */
static float quantize(const float* x, int n, int8_t* q) {
    float largest = 0.f;
    for (int i = 0; i < n; i++) largest = fmaxf(largest, fabsf(x[i]));
    if (largest == 0.f) {
        memset(q, 0, (size_t)n);
        return 0.f;
    }
    const float inverse = 127.f / largest;
    for (int i = 0; i < n; i++) q[i] = (int8_t)lrintf(x[i] * inverse);
    return largest / 127.f;
}

static float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

/* Hidden units of one LSTM layer: each unit's four gates (PyTorch's input, forget, cell, output) from [x; h],
   one weight row covering both halves and each half its own input scale; then the unit's cell and output, so
   the nonlinearity is split among the threads along with the products. */
static void lstm_units(const Tdt* m, int thread, int begin, int end) {
    const Work* w = m->work;
    const Linear* l = w->layer;
    (void)thread;
    for (int i = begin; i < end; i++) {
        const int8_t* rows[8];
        int32_t acc[8];
        float gates[4];
        for (int g = 0; g < 4; g++) {
            rows[g] = l->w + (size_t)(g * HIDDEN + i) * l->cols;
            rows[4 + g] = rows[g] + HIDDEN;
        }
        m->dot8(rows, w->xq, w->hq, HIDDEN, acc);
        for (int g = 0; g < 4; g++) {
            const int r = g * HIDDEN + i;
            gates[g] = (float)acc[g] * (w->sx * l->scale[r]) + (float)acc[4 + g] * (w->sh * l->scale[r]) + l->bias[r];
        }
        const float in = sigmoid(gates[0]), forget = sigmoid(gates[1]), cell = tanhf(gates[2]), out = sigmoid(gates[3]);
        w->c_out[i] = forget * w->c[i] + in * cell;
        w->h_out[i] = out * tanhf(w->c_out[i]);
    }
}

static void lstm_layer(Tdt* m, const Linear* l, const float* x, const float* h, const float* c,
                       float* h_out, float* c_out) {
    int8_t xq[HIDDEN], hq[HIDDEN];
    Work* w = m->work;
    w->layer = l, w->xq = xq, w->hq = hq, w->c = c, w->h_out = h_out, w->c_out = c_out;
    w->sx = quantize(x, HIDDEN, xq), w->sh = quantize(h, HIDDEN, hq);
    run(m, lstm_units, HIDDEN);
}

static void prediction_blocks(const Tdt* m, int thread, int begin, int end) {
    const Work* w = m->work;
    (void)thread;
    for (int b = begin; b < end; b++) {
        const int8_t* rows[8];
        int32_t acc[8];
        for (int k = 0; k < 8; k++) rows[k] = m->pred.w + (size_t)(8 * b + k) * HIDDEN;
        m->dot8(rows, w->pq, w->pq, HIDDEN, acc);
        for (int k = 0; k < 8; k++) {
            const int r = 8 * b + k;
            w->pred[r] = (float)acc[k] * (w->ps * m->pred.scale[r]) + m->pred.bias[r];
        }
    }
}

/* The prediction for *label* and the state it leads to. */
static void step(Tdt* m, int label, const State* from, float* pred, State* to) {
    const float* x = m->embed + (size_t)label * HIDDEN;
    lstm_layer(m, &m->lstm[0], x, from->h[0], from->c[0], to->h[0], to->c[0]);
    lstm_layer(m, &m->lstm[1], to->h[0], from->h[1], from->c[1], to->h[1], to->c[1]);
    int8_t hq[HIDDEN];
    Work* w = m->work;
    w->pq = hq, w->pred = pred;
    w->ps = quantize(to->h[1], HIDDEN, hq);
    run(m, prediction_blocks, HIDDEN / 8);
}

/* A thread's best token and duration per frame over the 8-row joint blocks it takes; each block stays in the
   core's first-level cache across the frames. A tie keeps the lower row. */
static void joint_blocks(const Tdt* m, int thread, int begin, int end) {
    Work* w = m->work;
    const Linear* l = &m->out;
    float* best_token = w->best_token[thread];
    float* best_duration = w->best_duration[thread];
    int* token = w->token[thread];
    int* duration = w->duration[thread];
    for (int b = begin; b < end; b++) {
        const int8_t* rows[8];
        for (int k = 0; k < 8; k++) rows[k] = l->w + (size_t)(8 * b + k) * HIDDEN;
        for (int f = 0; f < w->frames; f++) {
            int32_t acc[8];
            m->dot8(rows, w->q[f], w->q[f], HIDDEN, acc);
            for (int k = 0; k < 8 && 8 * b + k < OUTPUTS; k++) {
                const int r = 8 * b + k;
                const float logit = (float)acc[k] * (w->s[f] * l->scale[r]) + l->bias[r];
                if (r < TOKENS) {
                    if (logit > best_token[f] || (logit == best_token[f] && r < token[f])) best_token[f] = logit, token[f] = r;
                } else if (logit > best_duration[f] || (logit == best_duration[f] && r - TOKENS < duration[f])) {
                    best_duration[f] = logit, duration[f] = r - TOKENS;
                }
            }
        }
    }
}

/* The joint for *n* frames of *enc* against *pred*: the best token and duration index per frame, ties keeping
   the first index, as NumPy's argmax does. Each weight row is read once for all the frames. */
static void score(Tdt* m, const float* enc, int n, const float* pred, int* token, int* duration) {
    Work* w = m->work;
    w->frames = n;
    for (int f = 0; f < n; f++) {
        float joined[HIDDEN];
        for (int i = 0; i < HIDDEN; i++) joined[i] = fmaxf(enc[(size_t)f * HIDDEN + i] + pred[i], 0.f);
        w->s[f] = quantize(joined, HIDDEN, w->q[f]);
    }
    for (int t = 0; t < m->threads; t++)
        for (int f = 0; f < n; f++) {
            w->best_token[t][f] = w->best_duration[t][f] = -INFINITY;
            w->token[t][f] = TOKENS, w->duration[t][f] = DURATIONS;
        }
    run(m, joint_blocks, (OUTPUTS + 7) / 8);
    for (int f = 0; f < n; f++) {
        int bt = 0, bd = 0;
        for (int t = 1; t < m->threads; t++) {
            const float vt = w->best_token[t][f], vd = w->best_duration[t][f];
            if (vt > w->best_token[bt][f] || (vt == w->best_token[bt][f] && w->token[t][f] < w->token[bt][f])) bt = t;
            if (vd > w->best_duration[bd][f] ||
                (vd == w->best_duration[bd][f] && w->duration[t][f] < w->duration[bd][f]))
                bd = t;
        }
        token[f] = w->token[bt][f], duration[f] = w->duration[bd][f];
    }
}


/* ---- log-mel features -------------------------------------------------------------------------------- */

/* One 256-point complex FFT (radix 2, decimation in time) on input already in bit-reversed order. */
static void fft_scalar(const Mel* mel, float* re, float* im) {
    for (int len = 2; len <= HALF; len <<= 1) {
        const int half = len >> 1, stride = HALF / len;
        for (int i = 0; i < HALF; i += len)
            for (int j = 0; j < half; j++) {
                const float wr = mel->tw_re[j * stride], wi = mel->tw_im[j * stride];
                const float br = re[i + j + half], bi = im[i + j + half];
                const float vr = br * wr - bi * wi, vi = br * wi + bi * wr;
                const float ur = re[i + j], ui = im[i + j];
                re[i + j] = ur + vr, im[i + j] = ui + vi;
                re[i + j + half] = ur - vr, im[i + j + half] = ui - vi;
            }
    }
}

/* The same FFT on eight frames at once, one per lane: the very operations, so the same rounding. */
TARGET_AVX2 static void fft_avx2(const Mel* mel, __m256* re, __m256* im) {
    for (int len = 2; len <= HALF; len <<= 1) {
        const int half = len >> 1, stride = HALF / len;
        for (int i = 0; i < HALF; i += len)
            for (int j = 0; j < half; j++) {
                const __m256 wr = _mm256_set1_ps(mel->tw_re[j * stride]), wi = _mm256_set1_ps(mel->tw_im[j * stride]);
                const __m256 br = re[i + j + half], bi = im[i + j + half];
                const __m256 vr = _mm256_sub_ps(_mm256_mul_ps(br, wr), _mm256_mul_ps(bi, wi));
                const __m256 vi = _mm256_add_ps(_mm256_mul_ps(br, wi), _mm256_mul_ps(bi, wr));
                const __m256 ur = re[i + j], ui = im[i + j];
                re[i + j] = _mm256_add_ps(ur, vr), im[i + j] = _mm256_add_ps(ui, vi);
                re[i + j + half] = _mm256_sub_ps(ur, vr), im[i + j + half] = _mm256_sub_ps(ui, vi);
            }
    }
}

/* A frame's 257 power bins from its half-size transform Z: X[k] = E[k] + e^(-2 pi i k/512) O[k], where E and
   O are the even and odd samples' transforms, (Z[k] + conj Z[256-k]) / 2 and (Z[k] - conj Z[256-k]) / 2i. */
static void power_scalar(const Mel* mel, const float* zr, const float* zi, float* power) {
    for (int k = 0; k < BINS; k++) {
        const int a = k & (HALF - 1), b = (HALF - k) & (HALF - 1);
        const float er = 0.5f * (zr[a] + zr[b]), ei = 0.5f * (zi[a] - zi[b]);
        const float odd_r = 0.5f * (zi[a] + zi[b]), odd_i = -0.5f * (zr[a] - zr[b]);
        const float xr = er + (odd_r * mel->post_re[k] - odd_i * mel->post_im[k]);
        const float xi = ei + (odd_r * mel->post_im[k] + odd_i * mel->post_re[k]);
        power[k] = xr * xr + xi * xi;
    }
}

TARGET_AVX2 static void power_avx2(const Mel* mel, const __m256* zr, const __m256* zi, __m256* power) {
    const __m256 half = _mm256_set1_ps(0.5f), minus_half = _mm256_set1_ps(-0.5f);
    for (int k = 0; k < BINS; k++) {
        const int a = k & (HALF - 1), b = (HALF - k) & (HALF - 1);
        const __m256 wr = _mm256_set1_ps(mel->post_re[k]), wi = _mm256_set1_ps(mel->post_im[k]);
        const __m256 er = _mm256_mul_ps(half, _mm256_add_ps(zr[a], zr[b]));
        const __m256 ei = _mm256_mul_ps(half, _mm256_sub_ps(zi[a], zi[b]));
        const __m256 odd_r = _mm256_mul_ps(half, _mm256_add_ps(zi[a], zi[b]));
        const __m256 odd_i = _mm256_mul_ps(minus_half, _mm256_sub_ps(zr[a], zr[b]));
        const __m256 xr = _mm256_add_ps(er, _mm256_sub_ps(_mm256_mul_ps(odd_r, wr), _mm256_mul_ps(odd_i, wi)));
        const __m256 xi = _mm256_add_ps(ei, _mm256_add_ps(_mm256_mul_ps(odd_r, wi), _mm256_mul_ps(odd_i, wr)));
        power[k] = _mm256_add_ps(_mm256_mul_ps(xr, xr), _mm256_mul_ps(xi, xi));
    }
}

static float log_mel_band(const Mel* mel, const float* power, int band) {
    float sum = 0.f;
    for (int t = 0; t < mel->count[band]; t++) sum += power[mel->lo[band] + t] * mel->weights[mel->offset[band] + t];
    return logf(sum + LOG_GUARD);
}

static void mel_frame_scalar(const Tdt* m, int f) {
    const Mel* mel = m->mel;
    const Work* w = m->work;
    const float* base = w->padded + (size_t)f * HOP;
    float zr[HALF], zi[HALF], power[BINS];
    for (int n = 0; n < HALF; n++) {
        zr[mel->reverse[n]] = base[2 * n] * mel->window[2 * n];
        zi[mel->reverse[n]] = base[2 * n + 1] * mel->window[2 * n + 1];
    }
    fft_scalar(mel, zr, zi);
    power_scalar(mel, zr, zi, power);
    for (int b = 0; b < MELS; b++) w->log_mel[(size_t)f * MELS + b] = log_mel_band(mel, power, b);
}

TARGET_AVX2 static void mel_group_avx2(const Tdt* m, int group) {
    const Mel* mel = m->mel;
    const Work* w = m->work;
    __m256 zr[HALF], zi[HALF], power[BINS];
    float* lanes_r = (float*)zr;
    float* lanes_i = (float*)zi;
    for (int lane = 0; lane < LANES; lane++) {
        const int f = group * LANES + lane;
        const float* base = w->padded + (size_t)f * HOP;
        for (int n = 0; n < HALF; n++) {
            const int at = mel->reverse[n] * LANES + lane;
            lanes_r[at] = f < w->mel_frames ? base[2 * n] * mel->window[2 * n] : 0.f;
            lanes_i[at] = f < w->mel_frames ? base[2 * n + 1] * mel->window[2 * n + 1] : 0.f;
        }
    }
    fft_avx2(mel, zr, zi);
    power_avx2(mel, zr, zi, power);
    for (int lane = 0; lane < LANES; lane++) {
        const int f = group * LANES + lane;
        if (f >= w->mel_frames) break;
        float bins[BINS];
        for (int k = 0; k < BINS; k++) bins[k] = ((const float*)&power[k])[lane];
        for (int b = 0; b < MELS; b++) w->log_mel[(size_t)f * MELS + b] = log_mel_band(mel, bins, b);
    }
}

static void mel_groups(const Tdt* m, int thread, int begin, int end) {
    (void)thread;
    for (int g = begin; g < end; g++) {
        if (m->kernel >= KERNEL_AVX2) {
            mel_group_avx2(m, g);
            continue;
        }
        for (int f = g * LANES; f < (g + 1) * LANES && f < m->work->mel_frames; f++) mel_frame_scalar(m, f);
    }
}

/* Each band normalised over the frames: (x - mean) / (std + 1e-5), std with frames - 1, as NeMo does. */
static void mel_bands(const Tdt* m, int thread, int begin, int end) {
    const Work* w = m->work;
    const int frames = w->mel_frames;
    (void)thread;
    for (int b = begin; b < end; b++) {
        double sum = 0;
        for (int f = 0; f < frames; f++) sum += w->log_mel[(size_t)f * MELS + b];
        const float mean = (float)(sum / frames);
        double squares = 0;
        for (int f = 0; f < frames; f++) {
            const float d = w->log_mel[(size_t)f * MELS + b] - mean;
            squares += (double)d * d;
        }
        const float spread = (float)sqrt(squares / (frames - 1)) + 1e-5f;
        for (int f = 0; f < frames; f++) w->features[(size_t)b * frames + f] = (w->log_mel[(size_t)f * MELS + b] - mean) / spread;
    }
}

/* ---- the API --------------------------------------------------------------------------------------------- */

static void stop_workers(Tdt* m) {
    if (m->threads <= 1) return;
    add(&m->pool.stop, 1);
    add(&m->pool.generation, 1);
    wake_all(&m->pool.generation);
    for (int t = 1; t < m->threads; t++) {
        if (!m->workers[t]) continue;
#if defined(_MSC_VER)
        WaitForSingleObject(m->workers[t], INFINITE);
        CloseHandle(m->workers[t]);
#else
        pthread_join(m->workers[t], NULL);
#endif
    }
}

static int start_workers(Tdt* m) {
    for (int t = 1; t < m->threads; t++) {
        WorkerStart* start = (WorkerStart*)malloc(sizeof(WorkerStart));
        if (!start) return 0;
        start->m = m, start->thread = t;
#if defined(_MSC_VER)
        m->workers[t] = CreateThread(NULL, 0, worker, start, 0, NULL);
        if (!m->workers[t]) return free(start), 0;
#else
        if (pthread_create(&m->workers[t], NULL, worker, start) != 0) return free(start), 0;
#endif
    }
    return 1;
}

EXPORT void tdt_destroy(void* handle) {
    Tdt* m = (Tdt*)handle;
    if (!m) return;
    stop_workers(m);
    release(m->embed);
    free_linear(&m->lstm[0]), free_linear(&m->lstm[1]), free_linear(&m->pred), free_linear(&m->out);
    release(m->work);
    if (m->mel) free(m->mel->weights);
    free(m->mel);
    free(m);
}

/* The decoder, or NULL when *kernel* asks for instructions this CPU lacks, or memory runs out. *threads* share
   each product (1 to 16). LSTM rows are [weight_ih | weight_hh] (GATES x 2*HIDDEN), gates in PyTorch's order
   (input, forget, cell, output). */
EXPORT void* tdt_create(int kernel, int threads, const float* embed,
                        const int8_t* lstm0_w, const float* lstm0_scale, const float* lstm0_bias,
                        const int8_t* lstm1_w, const float* lstm1_scale, const float* lstm1_bias,
                        const int8_t* pred_w, const float* pred_scale, const float* pred_bias,
                        const int8_t* out_w, const float* out_scale, const float* out_bias) {
    const int best = best_kernel();
    if (kernel == KERNEL_AUTO) kernel = best;
    if (kernel > best || kernel < KERNEL_SCALAR || threads < 1 || threads > MAX_THREADS) return NULL;
    Tdt* m = (Tdt*)calloc(1, sizeof(Tdt));
    if (!m) return NULL;
    m->kernel = kernel, m->threads = threads;
    m->dot8 = kernel == KERNEL_VNNI_INT8 ? dot8_vnni_int8 : kernel == KERNEL_AVX2 ? dot8_avx2 : dot8_scalar;
    m->embed = (float*)aligned(sizeof(float) * TOKENS * HIDDEN);
    m->work = (Work*)aligned(sizeof(Work));
    int ok = m->embed != NULL && m->work != NULL;
    if (ok) memcpy(m->embed, embed, sizeof(float) * TOKENS * HIDDEN);
    ok = ok && copy_linear(&m->lstm[0], GATES, 2 * HIDDEN, lstm0_w, lstm0_scale, lstm0_bias);
    ok = ok && copy_linear(&m->lstm[1], GATES, 2 * HIDDEN, lstm1_w, lstm1_scale, lstm1_bias);
    ok = ok && copy_linear(&m->pred, HIDDEN, HIDDEN, pred_w, pred_scale, pred_bias);
    ok = ok && copy_linear(&m->out, OUTPUTS, HIDDEN, out_w, out_scale, out_bias);
    ok = ok && start_workers(m);
    if (ok) return m;
    tdt_destroy(m);
    return NULL;
}

EXPORT int tdt_kernel(const void* handle) { return ((const Tdt*)handle)->kernel; }

/* The window (N_FFT) and mel filters (MELS x BINS) for tdt_log_mel; 0 when memory runs out. */
EXPORT int tdt_set_mel(void* handle, const float* window, const float* filterbank) {
    Tdt* m = (Tdt*)handle;
    Mel* mel = (Mel*)calloc(1, sizeof(Mel));
    if (!mel) return 0;
    memcpy(mel->window, window, sizeof(mel->window));
    int total = 0;
    for (int b = 0; b < MELS; b++) {
        int lo = BINS, hi = -1;
        for (int k = 0; k < BINS; k++)
            if (filterbank[b * BINS + k] != 0.f) lo = k < lo ? k : lo, hi = k;
        mel->lo[b] = hi < 0 ? 0 : lo, mel->count[b] = hi < 0 ? 0 : hi - lo + 1, mel->offset[b] = total;
        total += mel->count[b];
    }
    mel->weights = (float*)malloc(sizeof(float) * (total ? total : 1));
    if (!mel->weights) return free(mel), 0;
    for (int b = 0; b < MELS; b++)
        memcpy(mel->weights + mel->offset[b], filterbank + b * BINS + mel->lo[b], sizeof(float) * mel->count[b]);
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < HALF / 2; i++) mel->tw_re[i] = (float)cos(2 * pi * i / HALF), mel->tw_im[i] = (float)-sin(2 * pi * i / HALF);
    for (int k = 0; k < BINS; k++) mel->post_re[k] = (float)cos(2 * pi * k / N_FFT), mel->post_im[k] = (float)-sin(2 * pi * k / N_FFT);
    for (int n = 0; n < HALF; n++) {
        int r = 0;
        for (int bit = 0; bit < 8; bit++) r |= ((n >> bit) & 1) << (7 - bit);
        mel->reverse[n] = r;
    }
    if (m->mel) free(m->mel->weights);
    free(m->mel);
    m->mel = mel;
    return 1;
}

/* The normalised log-mel features of *n* 16 kHz samples into *features* (MELS x frames, frames = 1 + n / HOP);
   the frame count, or -1 without tdt_set_mel, for fewer than N_FFT / 2 + 1 samples, or out of memory. */
EXPORT int tdt_log_mel(void* handle, const float* samples, int n, float* features) {
    Tdt* m = (Tdt*)handle;
    if (!m->mel || n <= N_FFT / 2) return -1;
    const int frames = 1 + n / HOP, pad = N_FFT / 2;
    float* padded = (float*)malloc(sizeof(float) * ((size_t)n + 2 * pad));
    float* log_mel = (float*)malloc(sizeof(float) * (size_t)frames * MELS);
    if (!padded || !log_mel) return free(padded), free(log_mel), -1;
    float* emphasised = padded + pad;
    emphasised[0] = samples[0];
    for (int i = 1; i < n; i++) emphasised[i] = samples[i] - PREEMPHASIS * samples[i - 1];
    for (int k = 1; k <= pad; k++) emphasised[-k] = emphasised[k], emphasised[n - 1 + k] = emphasised[n - 1 - k];
    Work* w = m->work;
    w->padded = padded, w->log_mel = log_mel, w->features = features, w->mel_frames = frames;
    run(m, mel_groups, (frames + LANES - 1) / LANES);
    run(m, mel_bands, MELS);
    free(padded), free(log_mel);
    return frames;
}


/* Token ids for *frames* joint-projected encoder frames (frames x HIDDEN), at most *capacity* of them; the
   count written, or -1 when *capacity* ran out (frames * MAX_SYMBOLS_PER_FRAME always suffices). One call
   at a time per decoder. */
EXPORT int tdt_decode(void* handle, const float* enc, int frames, int* tokens, int capacity) {
    Tdt* m = (Tdt*)handle;
    State* committed = (State*)calloc(2, sizeof(State));
    if (!committed) return -1;
    State* candidate = committed + 1;
    float pred[HIDDEN];
    int scored_token[WINDOW], scored_duration[WINDOW];
    int scored_from = -1, scored_count = 0, have_scores = 0, window = WINDOW, count = 0, t = 0;
    step(m, BLANK, committed, pred, candidate);
    while (t < frames) {
        const int frame_start = t;
        int symbols = 0, need_loop = 1;
        while (need_loop && symbols < MAX_SYMBOLS_PER_FRAME && t < frames) {
            if (!have_scores || t < scored_from || t >= scored_from + scored_count) {
                if (have_scores) window = window * 2 < WINDOW ? window * 2 : WINDOW;
                scored_count = window < frames - t ? window : frames - t;
                score(m, enc + (size_t)t * HIDDEN, scored_count, pred, scored_token, scored_duration);
                scored_from = t, have_scores = 1;
            }
            const int token = scored_token[t - scored_from], skip = kDurations[scored_duration[t - scored_from]];
            if (token == BLANK) {
                need_loop = 0;
            } else {
                if (count == capacity) {
                    free(committed);
                    return -1;
                }
                tokens[count++] = token;
                *committed = *candidate;
                step(m, token, committed, pred, candidate);
                have_scores = 0, window = FIRST_WINDOW;
            }
            symbols++;
            t += skip;
            need_loop = need_loop && skip == 0;
        }
        if (t <= frame_start) t = frame_start + 1;
    }
    free(committed);
    return count;
}
