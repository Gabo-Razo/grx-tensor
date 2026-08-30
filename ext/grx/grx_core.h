/*
 * grx_core.h — API publica del nucleo C de GRX
 * =============================================================
 * Compatible con compilacion universal y despacho dinamico SIMD
 * (AVX2 + FMA / SSE / Escalar C)
 * =============================================================
 */

#ifndef GRX_CORE_H
#define GRX_CORE_H

#include <stddef.h>

#ifdef _WIN32
  #define GRX_API __declspec(dllexport)
#else
  #define GRX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Diagnostico y nivel SIMD ----------------------------------- */
/* Retorna 2 para AVX2+FMA, 1 para SSE, 0 para Escalar portable C */
GRX_API int grx_simd_level(void);

/* ---- Memoria alineada -------------------------------------------- */
GRX_API double* grx_alloc(size_t n);
GRX_API void    grx_free(double *ptr);

/* ---- Element-wise aritmetica ------------------------------------- */
GRX_API void grx_add       (const double *a, const double *b, double *out, size_t n);
GRX_API void grx_sub       (const double *a, const double *b, double *out, size_t n);
GRX_API void grx_mul       (const double *a, const double *b, double *out, size_t n);
GRX_API void grx_div       (const double *a, const double *b, double *out, size_t n);
GRX_API void grx_scale     (const double *a, double s,        double *out, size_t n);
GRX_API void grx_negate    (const double *a,                  double *out, size_t n);
GRX_API void grx_add_scalar(const double *a, double s,        double *out, size_t n);

/* ---- Element-wise matematicas ------------------------------------ */
GRX_API void grx_abs    (const double *a, double *out, size_t n);
GRX_API void grx_sqrt   (const double *a, double *out, size_t n);
GRX_API void grx_log    (const double *a, double *out, size_t n);
GRX_API void grx_exp    (const double *a, double *out, size_t n);
GRX_API void grx_pow    (const double *a, double exp,  double *out, size_t n);
GRX_API void grx_clip   (const double *a, double lo, double hi, double *out, size_t n);
GRX_API void grx_square (const double *a, double *out, size_t n);

/* ---- Reducciones ------------------------------------------------- */
GRX_API double grx_sum (const double *a, size_t n);
GRX_API double grx_mean(const double *a, size_t n);
GRX_API double grx_max (const double *a, size_t n);
GRX_API double grx_min (const double *a, size_t n);

/* ---- Algebra lineal ---------------------------------------------- */
GRX_API double grx_dot    (const double *a, const double *b, size_t n);
GRX_API void   grx_matmul (const double *a, const double *b, double *out,
                            size_t M, size_t K, size_t N);

/* ---- Activaciones ------------------------------------------------ */
GRX_API void grx_relu        (const double *a, double *out, size_t n);
GRX_API void grx_leaky_relu  (const double *a, double alpha, double *out, size_t n);
GRX_API void grx_tanh_act    (const double *a, double *out, size_t n);
GRX_API void grx_sigmoid     (const double *a, double *out, size_t n);
GRX_API void grx_softmax     (const double *a, double *out, size_t n);

/* ---- Optimizadores (in-place) ------------------------------------ */
/* SGD: param -= lr * grad */
GRX_API void grx_sgd_step(double *param, const double *grad,
                           double lr, size_t n);

/* Adam: actualiza param, m, v in-place con aceleracion FMA */
GRX_API void grx_adam_step(double *param,
                            double *m, double *v,
                            const double *grad,
                            double lr, double beta1, double beta2,
                            double epsilon, double beta1t, double beta2t,
                            size_t n);

/* ---- Inicializacion de pesos ------------------------------------- */
/* Xavier uniform: U(-limit, limit), limit = sqrt(6 / (fan_in + fan_out)) */
GRX_API void grx_init_xavier_uniform(double *out, size_t n,
                                      size_t fan_in, size_t fan_out);
/* He normal: N(0, sqrt(2/fan_in)) */
GRX_API void grx_init_he_normal(double *out, size_t n, size_t fan_in);

/* Extension init for Ruby */
GRX_API void Init_grx_core(void);

#ifdef __cplusplus
}
#endif

#endif /* GRX_CORE_H */
