/*
 * grx_core.c — Núcleo C de GRX
 * =============================================================
 * Optimizaciones activas:
 *   - AVX2 + FMA: 4 doubles/ciclo con multiply-add fusionado
 *   - Loop unrolling x2: mayor ILP (Instruction Level Parallelism)
 *   - restrict: elimina alias analysis, permite más vectorización auto
 *   - Memoria alineada 32 bytes: habilita _mm256_load_pd (más rápido que loadu)
 *   - matmul con tiling: respeta líneas de caché L1 (64 bytes = 8 doubles)
 *   - Adam con FMA: beta*m + (1-beta)*grad en una pasada
 * =============================================================
 */

#define _USE_MATH_DEFINES   /* M_PI en Windows/MSVC */
#define _POSIX_C_SOURCE 200809L  /* posix_memalign, M_PI en glibc */
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

#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define GRX_AVX2_FMA 1
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define GRX_AVX2 1
#elif defined(__SSE2__)
  #include <emmintrin.h>
  #define GRX_SSE2 1
#endif

#define TILE 8

/* ================================================================
 * MEMORIA
 * ================================================================ */

GRX_API double* grx_alloc(size_t n) {
    if (__builtin_expect(n == 0, 0)) return NULL;
    void *ptr = NULL;
#if defined(_WIN32)
    ptr = _aligned_malloc(n * sizeof(double), 32);
#else
    if (posix_memalign(&ptr, 32, n * sizeof(double)) != 0) return NULL;
#endif
    return (double*)ptr;
}

GRX_API void grx_free(double *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ================================================================
 * MACROS SIMD INTERNOS
 * ================================================================ */

/* Carga/store: usa aligned si tenemos AVX2+FMA (memoria siempre alineada a 32b) */
#ifdef GRX_AVX2_FMA
  #define VLD(p)      _mm256_load_pd(p)
  #define VST(p, v)   _mm256_store_pd(p, v)
#elif defined(GRX_AVX2)
  #define VLD(p)      _mm256_loadu_pd(p)
  #define VST(p, v)   _mm256_storeu_pd(p, v)
#endif

/* ================================================================
 * ELEMENT-WISE ARITMÉTICA
 * ================================================================ */

#define BINOP_BODY(op_avx, op_scalar)                                   \
    size_t i = 0;                                                        \
    for (; i + 8 <= n; i += 8) {                                        \
        VST(out+i,   op_avx(VLD(a+i),   VLD(b+i)));                    \
        VST(out+i+4, op_avx(VLD(a+i+4), VLD(b+i+4)));                  \
    }                                                                    \
    for (; i + 4 <= n; i += 4) VST(out+i, op_avx(VLD(a+i), VLD(b+i)));\
    for (; i < n; i++) out[i] = op_scalar(a[i], b[i]);

#define SCALAR_ADD(x,y) ((x)+(y))
#define SCALAR_SUB(x,y) ((x)-(y))
#define SCALAR_MUL(x,y) ((x)*(y))
#define SCALAR_DIV(x,y) ((x)/(y))

GRX_API void grx_add(const double * restrict a, const double * restrict b,
                     double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    BINOP_BODY(_mm256_add_pd, SCALAR_ADD)
#else
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
#endif
}

GRX_API void grx_sub(const double * restrict a, const double * restrict b,
                     double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    BINOP_BODY(_mm256_sub_pd, SCALAR_SUB)
#else
    for (size_t i = 0; i < n; i++) out[i] = a[i] - b[i];
#endif
}

GRX_API void grx_mul(const double * restrict a, const double * restrict b,
                     double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    BINOP_BODY(_mm256_mul_pd, SCALAR_MUL)
#else
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
#endif
}

GRX_API void grx_div(const double * restrict a, const double * restrict b,
                     double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    BINOP_BODY(_mm256_div_pd, SCALAR_DIV)
#else
    for (size_t i = 0; i < n; i++) out[i] = a[i] / b[i];
#endif
}

GRX_API void grx_scale(const double * restrict a, double s,
                       double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    __m256d vs = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        VST(out+i,   _mm256_mul_pd(VLD(a+i),   vs));
        VST(out+i+4, _mm256_mul_pd(VLD(a+i+4), vs));
    }
    for (; i + 4 <= n; i += 4) VST(out+i, _mm256_mul_pd(VLD(a+i), vs));
    for (; i < n; i++) out[i] = a[i] * s;
#else
    for (size_t i = 0; i < n; i++) out[i] = a[i] * s;
#endif
}

GRX_API void grx_add_scalar(const double * restrict a, double s,
                             double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    __m256d vs = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        VST(out+i,   _mm256_add_pd(VLD(a+i),   vs));
        VST(out+i+4, _mm256_add_pd(VLD(a+i+4), vs));
    }
    for (; i + 4 <= n; i += 4) VST(out+i, _mm256_add_pd(VLD(a+i), vs));
    for (; i < n; i++) out[i] = a[i] + s;
#else
    for (size_t i = 0; i < n; i++) out[i] = a[i] + s;
#endif
}

GRX_API void grx_negate(const double * restrict a, double * restrict out, size_t n) {
    grx_scale(a, -1.0, out, n);
}

/* ================================================================
 * MATEMÁTICAS ELEMENT-WISE
 * ================================================================ */

GRX_API void grx_abs(const double * restrict a, double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    /* Máscara para limpiar el bit de signo (AND con 0x7FFFFFFFFFFFFFFF) */
    __m256d mask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        VST(out+i, _mm256_and_pd(VLD(a+i), mask));
    for (; i < n; i++) out[i] = fabs(a[i]);
#else
    for (size_t i = 0; i < n; i++) out[i] = fabs(a[i]);
#endif
}

GRX_API void grx_sqrt(const double * restrict a, double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        VST(out+i, _mm256_sqrt_pd(VLD(a+i)));
    for (; i < n; i++) out[i] = sqrt(a[i]);
#else
    for (size_t i = 0; i < n; i++) out[i] = sqrt(a[i]);
#endif
}

GRX_API void grx_square(const double * restrict a, double * restrict out, size_t n) {
    grx_mul(a, a, out, n);
}

GRX_API void grx_log(const double * restrict a, double * restrict out, size_t n) {
    /* log no tiene intrínseco SIMD estándar; -ffast-math + -march=native
     * permite al compilador auto-vectorizar con SVML si está disponible */
    for (size_t i = 0; i < n; i++) out[i] = log(a[i]);
}

GRX_API void grx_exp(const double * restrict a, double * restrict out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = exp(a[i]);
}

GRX_API void grx_pow(const double * restrict a, double e,
                     double * restrict out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = pow(a[i], e);
}

GRX_API void grx_clip(const double * restrict a, double lo, double hi,
                      double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    __m256d vlo = _mm256_set1_pd(lo);
    __m256d vhi = _mm256_set1_pd(hi);
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        VST(out+i, _mm256_min_pd(_mm256_max_pd(VLD(a+i), vlo), vhi));
    for (; i < n; i++) out[i] = a[i] < lo ? lo : (a[i] > hi ? hi : a[i]);
#else
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] < lo ? lo : (a[i] > hi ? hi : a[i]);
#endif
}

/* ================================================================
 * REDUCCIONES
 * ================================================================ */

GRX_API double grx_sum(const double * restrict a, size_t n) {
    double acc = 0.0;
#ifdef GRX_AVX2_FMA
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        v0 = _mm256_add_pd(v0, VLD(a+i));
        v1 = _mm256_add_pd(v1, VLD(a+i+4));
    }
    v0 = _mm256_add_pd(v0, v1);
    for (; i + 4 <= n; i += 4) v0 = _mm256_add_pd(v0, VLD(a+i));
    double tmp[4]; _mm256_store_pd(tmp, v0);
    acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) acc += a[i];
#elif defined(GRX_AVX2)
    __m256d vacc = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) vacc = _mm256_add_pd(vacc, VLD(a+i));
    double tmp[4]; _mm256_storeu_pd(tmp, vacc);
    acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) acc += a[i];
#else
    for (size_t i = 0; i < n; i++) acc += a[i];
#endif
    return acc;
}

GRX_API double grx_mean(const double * restrict a, size_t n) {
    return n > 0 ? grx_sum(a, n) / (double)n : 0.0;
}

GRX_API double grx_max(const double * restrict a, size_t n) {
    if (n == 0) return -DBL_MAX;
    double m = a[0];
    for (size_t i = 1; i < n; i++) if (a[i] > m) m = a[i];
    return m;
}

GRX_API double grx_min(const double * restrict a, size_t n) {
    if (n == 0) return DBL_MAX;
    double m = a[0];
    for (size_t i = 1; i < n; i++) if (a[i] < m) m = a[i];
    return m;
}

/* ================================================================
 * ÁLGEBRA LINEAL
 * ================================================================ */

GRX_API double grx_dot(const double * restrict a, const double * restrict b, size_t n) {
    double acc = 0.0;
#ifdef GRX_AVX2_FMA
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        v0 = _mm256_fmadd_pd(VLD(a+i),   VLD(b+i),   v0);
        v1 = _mm256_fmadd_pd(VLD(a+i+4), VLD(b+i+4), v1);
    }
    v0 = _mm256_add_pd(v0, v1);
    for (; i + 4 <= n; i += 4) v0 = _mm256_fmadd_pd(VLD(a+i), VLD(b+i), v0);
    double tmp[4]; _mm256_store_pd(tmp, v0);
    acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) acc += a[i] * b[i];
#elif defined(GRX_AVX2)
    __m256d vacc = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        vacc = _mm256_add_pd(vacc, _mm256_mul_pd(VLD(a+i), VLD(b+i)));
    double tmp[4]; _mm256_storeu_pd(tmp, vacc);
    acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) acc += a[i] * b[i];
#else
    for (size_t i = 0; i < n; i++) acc += a[i] * b[i];
#endif
    return acc;
}

/* matmul con tiling cache-friendly */
GRX_API void grx_matmul(const double * restrict a, const double * restrict b,
                        double * restrict out, size_t M, size_t K, size_t N) {
    memset(out, 0, M * N * sizeof(double));
    for (size_t ii = 0; ii < M; ii += TILE) {
        size_t ie = ii + TILE < M ? ii + TILE : M;
        for (size_t kk = 0; kk < K; kk += TILE) {
            size_t ke = kk + TILE < K ? kk + TILE : K;
            for (size_t jj = 0; jj < N; jj += TILE) {
                size_t je = jj + TILE < N ? jj + TILE : N;
                for (size_t i = ii; i < ie; i++)
                    for (size_t k = kk; k < ke; k++) {
                        double aik = a[i*K+k];
                        for (size_t j = jj; j < je; j++)
                            out[i*N+j] += aik * b[k*N+j];
                    }
            }
        }
    }
}

/* ================================================================
 * ACTIVACIONES
 * ================================================================ */

GRX_API void grx_relu(const double * restrict a, double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    __m256d vz = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        VST(out+i,   _mm256_max_pd(VLD(a+i),   vz));
        VST(out+i+4, _mm256_max_pd(VLD(a+i+4), vz));
    }
    for (; i + 4 <= n; i += 4) VST(out+i, _mm256_max_pd(VLD(a+i), vz));
    for (; i < n; i++) out[i] = a[i] > 0.0 ? a[i] : 0.0;
#else
    for (size_t i = 0; i < n; i++) out[i] = a[i] > 0.0 ? a[i] : 0.0;
#endif
}

GRX_API void grx_leaky_relu(const double * restrict a, double alpha,
                             double * restrict out, size_t n) {
#if defined(GRX_AVX2_FMA) || defined(GRX_AVX2)
    __m256d va = _mm256_set1_pd(alpha);
    __m256d vz = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = VLD(a+i);
        /* max(v, alpha*v): si v>0 → v, si v<=0 → alpha*v */
        VST(out+i, _mm256_blendv_pd(_mm256_mul_pd(v, va), v,
                                     _mm256_cmp_pd(v, vz, _CMP_GT_OQ)));
    }
    for (; i < n; i++) out[i] = a[i] > 0.0 ? a[i] : alpha * a[i];
#else
    for (size_t i = 0; i < n; i++) out[i] = a[i] > 0.0 ? a[i] : alpha * a[i];
#endif
}

GRX_API void grx_tanh_act(const double * restrict a, double * restrict out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = tanh(a[i]);
}

GRX_API void grx_sigmoid(const double * restrict a, double * restrict out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = 1.0 / (1.0 + exp(-a[i]));
}

GRX_API void grx_softmax(const double * restrict a, double * restrict out, size_t n) {
    double max_val = grx_max(a, n);
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) { out[i] = exp(a[i] - max_val); sum += out[i]; }
    double inv = 1.0 / sum;
    for (size_t i = 0; i < n; i++) out[i] *= inv;
}

/* ================================================================
 * OPTIMIZADORES (in-place sobre parámetros)
 * ================================================================ */

/* SGD: param[i] -= lr * grad[i] */
GRX_API void grx_sgd_step(double * restrict param, const double * restrict grad,
                           double lr, size_t n) {
#ifdef GRX_AVX2_FMA
    __m256d vlr = _mm256_set1_pd(lr);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        /* param -= lr * grad  usando FMA: param = -lr*grad + param */
        VST(param+i,   _mm256_fnmadd_pd(vlr, VLD(grad+i),   VLD(param+i)));
        VST(param+i+4, _mm256_fnmadd_pd(vlr, VLD(grad+i+4), VLD(param+i+4)));
    }
    for (; i + 4 <= n; i += 4)
        VST(param+i, _mm256_fnmadd_pd(vlr, VLD(grad+i), VLD(param+i)));
    for (; i < n; i++) param[i] -= lr * grad[i];
#else
    for (size_t i = 0; i < n; i++) param[i] -= lr * grad[i];
#endif
}

/*
 * Adam: Kingma & Ba 2015
 *   m = beta1*m + (1-beta1)*grad
 *   v = beta2*v + (1-beta2)*grad^2
 *   m_hat = m / (1 - beta1^t)
 *   v_hat = v / (1 - beta2^t)
 *   param -= lr * m_hat / (sqrt(v_hat) + eps)
 *
 * beta1t = beta1^t (pasado desde Ruby, se actualiza por paso)
 * beta2t = beta2^t
 */
GRX_API void grx_adam_step(double * restrict param,
                            double * restrict m, double * restrict v,
                            const double * restrict grad,
                            double lr, double beta1, double beta2,
                            double epsilon, double beta1t, double beta2t,
                            size_t n) {
    double one_m_b1 = 1.0 - beta1;
    double one_m_b2 = 1.0 - beta2;
    double inv_1mb1t = 1.0 / (1.0 - beta1t);
    double inv_1mb2t = 1.0 / (1.0 - beta2t);

#ifdef GRX_AVX2_FMA
    __m256d vb1    = _mm256_set1_pd(beta1);
    __m256d vb2    = _mm256_set1_pd(beta2);
    __m256d v1mb1  = _mm256_set1_pd(one_m_b1);
    __m256d v1mb2  = _mm256_set1_pd(one_m_b2);
    __m256d vlr    = _mm256_set1_pd(lr);
    __m256d veps   = _mm256_set1_pd(epsilon);
    __m256d vi1b1t = _mm256_set1_pd(inv_1mb1t);
    __m256d vi2b2t = _mm256_set1_pd(inv_1mb2t);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d g  = VLD(grad+i);
        /* m = beta1*m + (1-beta1)*g */
        __m256d mi = _mm256_fmadd_pd(vb1, VLD(m+i), _mm256_mul_pd(v1mb1, g));
        /* v = beta2*v + (1-beta2)*g^2 */
        __m256d vi = _mm256_fmadd_pd(vb2, VLD(v+i),
                         _mm256_mul_pd(v1mb2, _mm256_mul_pd(g, g)));
        VST(m+i, mi);
        VST(v+i, vi);
        /* m_hat = m / (1-beta1^t),  v_hat = v / (1-beta2^t) */
        __m256d mh = _mm256_mul_pd(mi, vi1b1t);
        __m256d vh = _mm256_mul_pd(vi, vi2b2t);
        /* param -= lr * mh / (sqrt(vh) + eps) */
        __m256d denom = _mm256_add_pd(_mm256_sqrt_pd(vh), veps);
        VST(param+i, _mm256_fnmadd_pd(vlr, _mm256_div_pd(mh, denom), VLD(param+i)));
    }
    for (; i < n; i++) {
        m[i] = beta1 * m[i] + one_m_b1 * grad[i];
        v[i] = beta2 * v[i] + one_m_b2 * grad[i] * grad[i];
        double mh = m[i] * inv_1mb1t;
        double vh = v[i] * inv_1mb2t;
        param[i] -= lr * mh / (sqrt(vh) + epsilon);
    }
#else
    for (size_t i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + one_m_b1 * grad[i];
        v[i] = beta2 * v[i] + one_m_b2 * grad[i] * grad[i];
        double mh = m[i] * inv_1mb1t;
        double vh = v[i] * inv_1mb2t;
        param[i] -= lr * mh / (sqrt(vh) + epsilon);
    }
#endif
}

/* ================================================================
 * INICIALIZACIÓN DE PESOS
 * ================================================================ */

/* LCG simple (no criptográfico, pero rápido y sin dependencias) */
static uint64_t grx_rng_state = 0;

static void grx_rng_seed(void) {
    grx_rng_state = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)&grx_rng_state;
}

/* Genera double uniforme en [0, 1) */
static double grx_rand01(void) {
    /* xorshift64 */
    grx_rng_state ^= grx_rng_state << 13;
    grx_rng_state ^= grx_rng_state >> 7;
    grx_rng_state ^= grx_rng_state << 17;
    return (double)(grx_rng_state >> 11) / (double)(1ULL << 53);
}

/* Box-Muller: genera par de normales N(0,1) */
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
    for (size_t i = 0; i < n; i++)
        out[i] = (grx_rand01() * 2.0 - 1.0) * limit;
}

GRX_API void grx_init_he_normal(double *out, size_t n, size_t fan_in) {
    grx_rng_seed();
    double std = sqrt(2.0 / (double)fan_in);
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

/* ============================================================
 * RUBY EXTENSION INIT — requerido por rake-compiler / mkmf
 * No hace nada: la librería se carga vía Fiddle, no como extensión Ruby nativa.
 * ============================================================ */
void Init_grx_core(void) { /* no-op */ }
