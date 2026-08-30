/*
 * grx_core.c — Nucleo C de GRX con Despacho Dinamico Multi-Target SIMD
 * ====================================================================
 * Caracteristicas de compatibilidad y rendimiento:
 *   - Compilacion base universal (sin banderas forzadas globales -mavx2)
 *   - Despacho en tiempo de ejecucion segun capacidades reales de la CPU:
 *       * AVX2 + FMA: 4 doubles/ciclo con multiply-add fusionado
 *       * Fallback escalar seguro: compatible con 100% de CPUs (x86_64, ARM, VMs)
 *   - Memoria alineada a 32 bytes: posix_memalign en Unix, _aligned_malloc en Windows
 *   - Matmul con cache tiling (L1 64 bytes)
 *   - Optimizadores in-place (SGD, Adam con correccion de sesgo)
 *   - Generadores de pesos xorshift64 y Box-Muller universales
 * ====================================================================
 */

#define _USE_MATH_DEFINES
#define _POSIX_C_SOURCE 200809L
#include "grx_core.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define GRX_TARGET_AVX2 __attribute__((target("avx2,fma")))
  #include <immintrin.h>
#else
  #define GRX_TARGET_AVX2
#endif

#define TILE 8

/* ================================================================
 * DETECCION DE CPU Y NIVEL SIMD
 * ================================================================ */

static int g_simd_level = -1;

static int grx_detect_simd(void) {
    if (__builtin_expect(g_simd_level != -1, 1)) {
        return g_simd_level;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        g_simd_level = 2; /* AVX2 + FMA */
        return 2;
    } else if (__builtin_cpu_supports("sse2")) {
        g_simd_level = 1; /* SSE2 */
        return 1;
    }
#endif

    g_simd_level = 0; /* Escalar universal */
    return 0;
}

GRX_API int grx_simd_level(void) {
    return grx_detect_simd();
}

/* ================================================================
 * GESTION DE MEMORIA
 * ================================================================ */

GRX_API double* grx_alloc(size_t n) {
    if (__builtin_expect(n == 0, 0)) return NULL;
    void *ptr = NULL;
#if defined(_WIN32) || defined(_WIN64)
    ptr = _aligned_malloc(n * sizeof(double), 32);
#else
    if (posix_memalign(&ptr, 32, n * sizeof(double)) != 0) {
        ptr = malloc(n * sizeof(double));
    }
#endif
    return (double*)ptr;
}

GRX_API void grx_free(double *ptr) {
    if (!ptr) return;
#if defined(_WIN32) || defined(_WIN64)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ================================================================
 * ARITMETICA ELEMENT-WISE: KERNELS AVX2 Y ESCALARES
 * ================================================================ */

#if defined(__GNUC__) || defined(__clang__)

GRX_TARGET_AVX2
static void grx_add_avx2(const double *a, const double *b, double *out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_pd(out + i,     _mm256_add_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i)));
        _mm256_storeu_pd(out + i + 4, _mm256_add_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(out + i, _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}

GRX_TARGET_AVX2
static void grx_sub_avx2(const double *a, const double *b, double *out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_pd(out + i,     _mm256_sub_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i)));
        _mm256_storeu_pd(out + i + 4, _mm256_sub_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(out + i, _mm256_sub_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
    }
    for (; i < n; i++) out[i] = a[i] - b[i];
}

GRX_TARGET_AVX2
static void grx_mul_avx2(const double *a, const double *b, double *out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_pd(out + i,     _mm256_mul_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i)));
        _mm256_storeu_pd(out + i + 4, _mm256_mul_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(out + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
}

GRX_TARGET_AVX2
static void grx_div_avx2(const double *a, const double *b, double *out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_pd(out + i,     _mm256_div_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i)));
        _mm256_storeu_pd(out + i + 4, _mm256_div_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(out + i, _mm256_div_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
    }
    for (; i < n; i++) out[i] = a[i] / b[i];
}

GRX_TARGET_AVX2
static void grx_scale_avx2(const double *a, double s, double *out, size_t n) {
    __m256d vs = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_pd(out + i,     _mm256_mul_pd(_mm256_loadu_pd(a + i), vs));
        _mm256_storeu_pd(out + i + 4, _mm256_mul_pd(_mm256_loadu_pd(a + i + 4), vs));
    }
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(out + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), vs));
    }
    for (; i < n; i++) out[i] = a[i] * s;
}

GRX_TARGET_AVX2
static void grx_add_scalar_avx2(const double *a, double s, double *out, size_t n) {
    __m256d vs = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_pd(out + i,     _mm256_add_pd(_mm256_loadu_pd(a + i), vs));
        _mm256_storeu_pd(out + i + 4, _mm256_add_pd(_mm256_loadu_pd(a + i + 4), vs));
    }
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(out + i, _mm256_add_pd(_mm256_loadu_pd(a + i), vs));
    }
    for (; i < n; i++) out[i] = a[i] + s;
}

GRX_TARGET_AVX2
static double grx_sum_avx2(const double *a, size_t n) {
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        v0 = _mm256_add_pd(v0, _mm256_loadu_pd(a + i));
        v1 = _mm256_add_pd(v1, _mm256_loadu_pd(a + i + 4));
    }
    v0 = _mm256_add_pd(v0, v1);
    for (; i + 4 <= n; i += 4) {
        v0 = _mm256_add_pd(v0, _mm256_loadu_pd(a + i));
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, v0);
    double acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) acc += a[i];
    return acc;
}

GRX_TARGET_AVX2
static double grx_dot_avx2(const double *a, const double *b, size_t n) {
    __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i),     acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4), acc1);
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    for (; i + 4 <= n; i += 4) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), acc0);
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, acc0);
    double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

GRX_TARGET_AVX2
static void grx_relu_avx2(const double *a, double *out, size_t n) {
    __m256d zero = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_pd(out + i,     _mm256_max_pd(_mm256_loadu_pd(a + i), zero));
        _mm256_storeu_pd(out + i + 4, _mm256_max_pd(_mm256_loadu_pd(a + i + 4), zero));
    }
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(out + i, _mm256_max_pd(_mm256_loadu_pd(a + i), zero));
    }
    for (; i < n; i++) out[i] = a[i] > 0.0 ? a[i] : 0.0;
}

GRX_TARGET_AVX2
static void grx_adam_step_avx2(double *param, double *m, double *v,
                               const double *grad, double lr,
                               double beta1, double beta2, double epsilon,
                               double beta1t, double beta2t, size_t n) {
    __m256d vb1   = _mm256_set1_pd(beta1);
    __m256d vb2   = _mm256_set1_pd(beta2);
    __m256d vom1  = _mm256_set1_pd(1.0 - beta1);
    __m256d vom2  = _mm256_set1_pd(1.0 - beta2);
    __m256d vcb1  = _mm256_set1_pd(1.0 / (1.0 - beta1t));
    __m256d vcb2  = _mm256_set1_pd(1.0 / (1.0 - beta2t));
    __m256d vlr   = _mm256_set1_pd(lr);
    __m256d veps  = _mm256_set1_pd(epsilon);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d g   = _mm256_loadu_pd(grad + i);
        __m256d mi  = _mm256_loadu_pd(m + i);
        __m256d vi  = _mm256_loadu_pd(v + i);
        __m256d p   = _mm256_loadu_pd(param + i);

        mi = _mm256_fmadd_pd(vb1, mi, _mm256_mul_pd(vom1, g));
        vi = _mm256_fmadd_pd(vb2, vi, _mm256_mul_pd(vom2, _mm256_mul_pd(g, g)));

        _mm256_storeu_pd(m + i, mi);
        _mm256_storeu_pd(v + i, vi);

        __m256d m_hat = _mm256_mul_pd(mi, vcb1);
        __m256d v_hat = _mm256_mul_pd(vi, vcb2);
        __m256d denom = _mm256_add_pd(_mm256_sqrt_pd(v_hat), veps);
        __m256d step  = _mm256_div_pd(_mm256_mul_pd(vlr, m_hat), denom);

        _mm256_storeu_pd(param + i, _mm256_sub_pd(p, step));
    }

    double cb1 = 1.0 / (1.0 - beta1t);
    double cb2 = 1.0 / (1.0 - beta2t);
    for (; i < n; i++) {
        double g = grad[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        double m_hat = m[i] * cb1;
        double v_hat = v[i] * cb2;
        param[i] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

#endif /* GCC / Clang */

/* ================================================================
 * FUNCIONES PUBLICAS CON DESPACHO DINAMICO
 * ================================================================ */

GRX_API void grx_add(const double *a, const double *b, double *out, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        grx_add_avx2(a, b, out, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

GRX_API void grx_sub(const double *a, const double *b, double *out, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        grx_sub_avx2(a, b, out, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) out[i] = a[i] - b[i];
}

GRX_API void grx_mul(const double *a, const double *b, double *out, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        grx_mul_avx2(a, b, out, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
}

GRX_API void grx_div(const double *a, const double *b, double *out, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        grx_div_avx2(a, b, out, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) out[i] = a[i] / b[i];
}

GRX_API void grx_scale(const double *a, double s, double *out, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        grx_scale_avx2(a, s, out, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) out[i] = a[i] * s;
}

GRX_API void grx_add_scalar(const double *a, double s, double *out, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        grx_add_scalar_avx2(a, s, out, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) out[i] = a[i] + s;
}

GRX_API void grx_negate(const double *a, double *out, size_t n) {
    grx_scale(a, -1.0, out, n);
}

/* ================================================================
 * MATEMATICAS ELEMENT-WISE
 * ================================================================ */

GRX_API void grx_abs(const double *a, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = fabs(a[i]);
}

GRX_API void grx_sqrt(const double *a, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = sqrt(a[i]);
}

GRX_API void grx_square(const double *a, double *out, size_t n) {
    grx_mul(a, a, out, n);
}

GRX_API void grx_log(const double *a, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = log(a[i]);
}

GRX_API void grx_exp(const double *a, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = exp(a[i]);
}

GRX_API void grx_pow(const double *a, double e, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = pow(a[i], e);
}

GRX_API void grx_clip(const double *a, double lo, double hi, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double v = a[i];
        out[i] = (v < lo) ? lo : ((v > hi) ? hi : v);
    }
}

/* ================================================================
 * REDUCCIONES
 * ================================================================ */

GRX_API double grx_sum(const double *a, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        return grx_sum_avx2(a, n);
    }
#endif
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) acc += a[i];
    return acc;
}

GRX_API double grx_mean(const double *a, size_t n) {
    if (n == 0) return 0.0;
    return grx_sum(a, n) / (double)n;
}

GRX_API double grx_max(const double *a, size_t n) {
    if (n == 0) return 0.0;
    double m = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] > m) m = a[i];
    }
    return m;
}

GRX_API double grx_min(const double *a, size_t n) {
    if (n == 0) return 0.0;
    double m = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] < m) m = a[i];
    }
    return m;
}

/* ================================================================
 * ALGEBRA LINEAL
 * ================================================================ */

GRX_API double grx_dot(const double *a, const double *b, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        return grx_dot_avx2(a, b, n);
    }
#endif
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

GRX_API void grx_matmul(const double *a, const double *b, double *out,
                        size_t M, size_t K, size_t N) {
    memset(out, 0, M * N * sizeof(double));

    /* Multiplicacion de matrices optimizada por bloques con cache tiling */
    for (size_t bi = 0; bi < M; bi += TILE) {
        size_t imax = bi + TILE < M ? bi + TILE : M;
        for (size_t bk = 0; bk < K; bk += TILE) {
            size_t kmax = bk + TILE < K ? bk + TILE : K;
            for (size_t bj = 0; bj < N; bj += TILE) {
                size_t jmax = bj + TILE < N ? bj + TILE : N;

                for (size_t i = bi; i < imax; i++) {
                    for (size_t k = bk; k < kmax; k++) {
                        double a_ik = a[i * K + k];
                        double *out_i = out + i * N;
                        const double *b_k = b + k * N;
                        for (size_t j = bj; j < jmax; j++) {
                            out_i[j] += a_ik * b_k[j];
                        }
                    }
                }
            }
        }
    }
}

/* ================================================================
 * ACTIVACIONES
 * ================================================================ */

GRX_API void grx_relu(const double *a, double *out, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        grx_relu_avx2(a, out, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) out[i] = a[i] > 0.0 ? a[i] : 0.0;
}

GRX_API void grx_leaky_relu(const double *a, double alpha, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double v = a[i];
        out[i] = v >= 0.0 ? v : alpha * v;
    }
}

GRX_API void grx_tanh_act(const double *a, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = tanh(a[i]);
}

GRX_API void grx_sigmoid(const double *a, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double v = a[i];
        if (v >= 0.0) {
            double ev = exp(-v);
            out[i] = 1.0 / (1.0 + ev);
        } else {
            double ev = exp(v);
            out[i] = ev / (1.0 + ev);
        }
    }
}

GRX_API void grx_softmax(const double *a, double *out, size_t n) {
    if (n == 0) return;
    double max_v = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] > max_v) max_v = a[i];
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double ev = exp(a[i] - max_v);
        out[i] = ev;
        sum += ev;
    }
    double inv = 1.0 / sum;
    for (size_t i = 0; i < n; i++) out[i] *= inv;
}

/* ================================================================
 * OPTIMIZADORES IN-PLACE
 * ================================================================ */

GRX_API void grx_sgd_step(double *param, const double *grad,
                          double lr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        param[i] -= lr * grad[i];
    }
}

GRX_API void grx_adam_step(double *param, double *m, double *v,
                           const double *grad, double lr,
                           double beta1, double beta2, double epsilon,
                           double beta1t, double beta2t, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    if (grx_detect_simd() >= 2) {
        grx_adam_step_avx2(param, m, v, grad, lr, beta1, beta2, epsilon, beta1t, beta2t, n);
        return;
    }
#endif
    double cb1 = 1.0 / (1.0 - beta1t);
    double cb2 = 1.0 / (1.0 - beta2t);
    for (size_t i = 0; i < n; i++) {
        double g = grad[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        double m_hat = m[i] * cb1;
        double v_hat = v[i] * cb2;
        param[i] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

/* ================================================================
 * INICIALIZACION DE PESOS
 * ================================================================ */

static uint64_t grx_rng_state = 0;

static void grx_rng_seed(void) {
    if (grx_rng_state == 0) {
        grx_rng_state = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)&grx_rng_state ^ 0x853c49e6748fea9bULL;
        if (grx_rng_state == 0) grx_rng_state = 1;
    }
}

/* Generador xorshift64* puramente escalar y universal */
static double grx_rand01(void) {
    grx_rng_state ^= grx_rng_state << 13;
    grx_rng_state ^= grx_rng_state >> 7;
    grx_rng_state ^= grx_rng_state << 17;
    uint64_t val = grx_rng_state * 0x2545F4914F6CDD1DULL;
    return (double)(val >> 11) * (1.0 / 9007199254740992.0); /* 2^53 */
}

static void grx_box_muller(double *z0, double *z1) {
    double u1, u2;
    do { u1 = grx_rand01(); } while (u1 < 1e-15);
    u2 = grx_rand01();
    double r = sqrt(-2.0 * log(u1));
    *z0 = r * cos(2.0 * M_PI * u2);
    *z1 = r * sin(2.0 * M_PI * u2);
}

GRX_API void grx_init_xavier_uniform(double *out, size_t n,
                                     size_t fan_in, size_t fan_out) {
    grx_rng_seed();
    double limit = sqrt(6.0 / (double)(fan_in + fan_out));
    for (size_t i = 0; i < n; i++) {
        out[i] = (grx_rand01() * 2.0 - 1.0) * limit;
    }
}

GRX_API void grx_init_he_normal(double *out, size_t n, size_t fan_in) {
    grx_rng_seed();
    double std = sqrt(2.0 / (double)(fan_in > 0 ? fan_in : 1));
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        double z0, z1;
        grx_box_muller(&z0, &z1);
        out[i]   = z0 * std;
        out[i+1] = z1 * std;
    }
    if (i < n) {
        double z0, z1;
        grx_box_muller(&z0, &z1);
        out[i] = z0 * std;
    }
}

/* ================================================================
 * RUBY EXTENSION ENTRYPOINT
 * ================================================================ */
GRX_API void Init_grx_core(void) {
    grx_detect_simd();
}
