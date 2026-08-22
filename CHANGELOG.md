# Changelog

All notable changes to GRX-Tensor are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- OpenMP parallelization for element-wise operations
- BLAS (`cblas_dgemm`) for production-grade matrix multiplication
- Broadcasting — automatic shape expansion
- `float32` support (8 values/cycle with AVX2)
- Move autograd graph to C to eliminate Ruby GC overhead
- `Conv2d`, `LSTM`, `MultiheadAttention` layers
- CUDA extension (`grx-tensor-cuda`)

---

## [0.2.0] - 2026-08-21

### Added
- **Differentiable Reductions & Operations**: `Tensor#sum` and `Tensor#mean` now create autograd DAG nodes supporting direct `loss.backward`.
- **Differentiable Loss Functions**: `MSELoss`, `MAELoss`, `BCELoss`, `CrossEntropyLoss`, and `HuberLoss` now return differentiable `Tensor` instances.
- **Advanced Neural Layers**:
  - `GRX::NN::Embedding` — trainable dense token embeddings with backpropagation.
  - `GRX::NN::LayerNorm` — feature normalization with trainable gamma/beta parameters.
- **Model Serialization**: High-speed binary `.grx` format with `GRX1` magic header for instant weight saving and loading (`Module#save_weights`, `Module#load_weights`).
- **Data Pipelines**: `GRX::Data::TensorDataset` and `GRX::Data::DataLoader` for mini-batching, shuffling, and dataset iteration.
- **Gradient Clipping**: `GRX::Utils.clip_grad_norm!` for stabilizing deep networks and recurrent architectures.
- **Robust Windows DLL Resolution**: Multi-path search for `grx_core.dll` across multiple fallback directories.
- **Helper methods**: `GRX.mode` and `GRX.c_loaded?`.
- **Spanish Documentation**: Complete `README.es.md` and `GUIA_PRINCIPIANTES.md`.

---

## [0.1.0] - 2026-05-11

### Added

**C kernel (`ext/grx/grx_core.c`)**
- Memory management: `grx_alloc` / `grx_free` — 32-byte aligned allocation via `posix_memalign` (Linux/macOS) and `_aligned_malloc` (Windows), required for AVX2 `_mm256_load_pd`
- Element-wise arithmetic: `grx_add`, `grx_sub`, `grx_mul`, `grx_div`, `grx_scale`, `grx_add_scalar`, `grx_negate` — AVX2+FMA with 2× loop unrolling
- Math ops: `grx_abs`, `grx_sqrt`, `grx_square`, `grx_log`, `grx_exp`, `grx_pow`, `grx_clip`
- Reductions: `grx_sum`, `grx_mean`, `grx_max`, `grx_min`
- Linear algebra: `grx_dot` (FMA with dual accumulators for ILP), `grx_matmul` (cache-friendly tiling, TILE=8)
- Activations: `grx_relu`, `grx_leaky_relu`, `grx_tanh_act`, `grx_sigmoid`, `grx_softmax`
- Optimizers: `grx_sgd_step` (FMA in-place), `grx_adam_step` (full Adam inner loop in C with FMA)
- Weight initialization: `grx_init_xavier_uniform`, `grx_init_he_normal` (Box-Muller)
- SIMD dispatch: AVX2+FMA → AVX2 → SSE2 → scalar, selected at compile time via `-march=native`

**Ruby layer**
- `GRX::Storage` — native memory buffer backed by `Fiddle::Pointer`; Ruby `Array` fallback when C is unavailable
- `GRX::Tensor` — shape, strides, offset (zero-copy `reshape` and `transpose`); all numeric ops delegate to C
- Autograd — topological BFS graph traversal; `backward_fn` closures for `+`, `-`, `*`, `/`, `square`, `sqrt`, `log`, `exp`, `relu`, `leaky_relu`, `tanh`, `sigmoid`, `matmul`
- `GRX::CAPI` — Fiddle bridge; detects platform and loads `libgrx_core.so` / `.dylib` / `.dll`
- `GRX::NN::Linear`, `Sequential`, `ReLU`, `LeakyReLU`, `Tanh`, `Sigmoid`, `Softmax`, `Dropout`, `BatchNorm1d`
- `GRX::Optim::SGD` (momentum, weight decay), `GRX::Optim::Adam` (bias correction, weight decay)
- `GRX::Loss::MSELoss`, `MAELoss`, `BCELoss`, `CrossEntropyLoss`, `HuberLoss`
- Factory helpers: `GRX.tensor`, `GRX.zeros`, `GRX.ones`, `GRX.rand`, `GRX.randn`, `Tensor.xavier_uniform`, `Tensor.he_normal`

**Build system**
- `ext/unix/Makefile` — compiles directly to `lib/grx/libgrx_core.so` (Linux) or `.dylib` (macOS); no intermediate file
- `ext/windows/Makefile.mingw` — compiles directly to `lib/grx/grx_core.dll` via MinGW-w64
- `ext/grx/extconf.rb` — rake-compiler config for `gem install` auto-compilation
- `.gitignore` — compiled binaries excluded from version control

**Tests**
- 43 tests, 10 121 assertions across `test_tensor.rb` and `test_nn.rb`
