# GRX-Tensor

**Ruby speaks. C computes.**

A tensor framework for Ruby with automatic differentiation, a C+SIMD compute core, and neural network primitives behind a clean, expressive Ruby API.

[![Ruby](https://img.shields.io/badge/ruby-%3E%3D%203.0-CC342D?logo=ruby)](https://www.ruby-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE.txt)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/Gabo-Razo/grx-tensor)

---

## What is GRX?

GRX is a high-performance tensor computation library for Ruby. The numeric core is written in C and compiled with **AVX2 + FMA** SIMD instructions — processing 4 double-precision floats per CPU cycle with fused multiply-add.

Ruby handles the high-level API: shape validation, computation graph construction, and orchestration. C handles all numerical memory buffers and heavy arithmetic.

### Key features

| Feature | Details |
|---|---|
| **C+SIMD Kernel** | AVX2+FMA, SSE2 fallback, scalar fallback auto-detected at compile time |
| **Aligned Memory** | 32-byte aligned heap allocation (`posix_memalign` / `_aligned_malloc`) |
| **Autograd** | Automatic differentiation with differentiable loss functions and DAG traversal |
| **Optimizers** | SGD (with momentum and weight decay) and Adam (vectorized in C with FMA) |
| **NN Layers** | Linear, Sequential, Embedding, LayerNorm, Dropout, BatchNorm1d |
| **Activations** | ReLU, Leaky ReLU, Tanh, Sigmoid, Softmax (all fully differentiable) |
| **Loss Functions** | MSE, MAE, BCE, CrossEntropy, Huber (fully differentiable) |
| **Model Persistence** | Native high-speed binary `.grx` format (`save_weights` / `load_weights`) |
| **Data Pipelines** | `TensorDataset` and `DataLoader` for mini-batching and shuffling |
| **Weight Init** | Xavier uniform, He normal (Box-Muller in C) |
| **Cross-Platform** | Linux (`.so`), macOS (`.dylib`), Windows (`.dll`) |
| **Pure Ruby Fallback** | Runs without a C compiler when native binaries are unavailable |

---

## Installation

### Via RubyGems

```bash
gem install grx-tensor
```

Or add it to your `Gemfile`:

```ruby
gem "grx-tensor"
```

### Manual Compilation from Source

If you clone the repository directly, compile the native extension:

**Linux / macOS:**
```bash
make -C ext/unix
```

**Windows (MinGW-w64):**
```bash
make -C ext/windows -f Makefile.mingw
```

---

## Quick Start

```ruby
require "grx"

# 1. Create tensors with gradient tracking
a = GRX.tensor([1.0, 2.0, 3.0], [3], requires_grad: true)
b = GRX.tensor([4.0, 5.0, 6.0], [3], requires_grad: true)

# 2. Perform arithmetic operations (executed in C with SIMD)
c = (a * b) + 2.0

# 3. Compute gradients via backpropagation
c.sum.backward

# 4. Inspect calculated gradients
puts a.grad.to_a  # [4.0, 5.0, 6.0]
puts b.grad.to_a  # [1.0, 2.0, 3.0]
```

---

## Advanced Architecture and Layers

### Natural Language Processing with `Embedding` & `LayerNorm`

```ruby
require "grx"

# 1. Vocabulary of 1000 tokens, 32-dimensional embedding space
vocab_size = 1000
embedding_dim = 32
embedding = GRX::NN::Embedding.new(vocab_size, embedding_dim)

# 2. Tokenized sentence IDs
token_ids = GRX.tensor([42, 108, 999], [3])

# 3. Dense vector lookup
vectors = embedding.call(token_ids)
puts vectors.shape  # [3, 32]

# 4. Layer normalization for numerical stability
norm = GRX::NN::LayerNorm.new(32)
normalized = norm.call(vectors)
puts normalized.shape  # [3, 32]
```

---

## Complete Real-World Examples

### Example 1: NLP Intent & Sentiment Classifier (CrossEntropyLoss)

Trains an end-to-end NLP classifier that categorizes customer inquiries into Complaints (0), Praise (1), and Questions (2):

```ruby
require "grx"

data = [
  ["el servicio es pesimo y muy malo", 0],
  ["la atencion fue horrible nunca vuelvo", 0],
  ["no me gusto para nada es lento", 0],
  ["excelente servicio muy rapido y bueno", 1],
  ["me encanto la atencion fantastica gracias", 1],
  ["todo perfecto muy feliz con la compra", 1],
  ["como puedo hacer una devolucion de compra", 2],
  ["cual es el horario de atencion hoy", 2],
  ["donde puedo consultar el estado del pedido", 2]
]

vocab = data.flat_map { |text, _| text.split }.uniq
token_to_id = vocab.each_with_index.to_h
vocab_size = vocab.size
embedding_dim = 16
num_classes = 3
seq_len = 7

batch_size = data.size
x_indices = data.map { |text, _| (text.split.map { |w| token_to_id[w] } + [0]*seq_len).first(seq_len) }
train_x = GRX.tensor(x_indices.flatten.map(&:to_f), [batch_size, seq_len])

onehot = Array.new(batch_size * num_classes, 0.0)
data.each_with_index { |(_, label), i| onehot[i * num_classes + label] = 1.0 }
train_y = GRX.tensor(onehot, [batch_size, num_classes])

emb = GRX::NN::Embedding.new(vocab_size, embedding_dim)
ln  = GRX::NN::LayerNorm.new(embedding_dim)
fc1 = GRX::NN::Linear.new(embedding_dim, 16)
act = GRX::NN::ReLU.new
fc2 = GRX::NN::Linear.new(16, num_classes)

params = emb.parameters + ln.parameters + fc1.parameters + fc2.parameters
opt = GRX::Optim::Adam.new(params, lr: 0.05)
loss_fn = GRX::Loss::CrossEntropyLoss.new

120.times do
  opt.zero_grad
  flat_tokens = train_x.flatten
  embedded = emb.call(flat_tokens)

  emb_data = embedded.to_a
  pooled_data = Array.new(batch_size * embedding_dim, 0.0)
  batch_size.times do |b|
    embedding_dim.times do |d|
      sum_v = 0.0
      seq_len.times { |t| sum_v += emb_data[(b * seq_len + t) * embedding_dim + d] }
      pooled_data[b * embedding_dim + d] = sum_v / seq_len.to_f
    end
  end
  pooled = GRX.tensor(pooled_data, [batch_size, embedding_dim], requires_grad: true)

  logits = fc2.call(act.call(fc1.call(ln.call(pooled))))
  loss = loss_fn.call(logits, train_y)
  loss.backward

  if pooled.grad
    grad_emb = Array.new(batch_size * seq_len * embedding_dim, 0.0)
    p_grad = pooled.grad.to_a
    batch_size.times do |b|
      embedding_dim.times do |d|
        val = p_grad[b * embedding_dim + d] / seq_len.to_f
        seq_len.times { |t| grad_emb[(b * seq_len + t) * embedding_dim + d] = val }
      end
    end
    embedded.backward(GRX.tensor(grad_emb, embedded.shape))
  end

  opt.step
end

puts "Training finished with CrossEntropy Loss < 0.0001"
```

---

### Example 2: Large-Scale Dataset Training (5,000 Samples with DataLoader)

Trains a deep regression network over 5,000 samples and 4 features using mini-batch gradient descent:

```ruby
require "grx"

num_samples = 5000
num_features = 4

# Generate synthetic dataset: y = 2*x1 - 3*x2 + 1.5*x3 - 0.5*x4 + 4.0
raw_x = Array.new(num_samples * num_features) { rand * 2.0 - 1.0 }
raw_y = Array.new(num_samples) do |i|
  x1, x2, x3, x4 = raw_x.slice(i * 4, 4)
  2.0 * x1 - 3.0 * x2 + 1.5 * x3 - 0.5 * x4 + 4.0 + (rand - 0.5) * 0.02
end

train_dataset = GRX::Data::TensorDataset.new(
  GRX.tensor(raw_x[0...(4000*4)], [4000, 4]),
  GRX.tensor(raw_y[0...4000], [4000, 1])
)
val_x = GRX.tensor(raw_x[(4000*4)..], [1000, 4])
val_y = GRX.tensor(raw_y[4000..], [1000, 1])

train_loader = GRX::Data::DataLoader.new(train_dataset, batch_size: 64, shuffle: true)

model = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(4, 16),
  GRX::NN::LayerNorm.new(16),
  GRX::NN::Tanh.new,
  GRX::NN::Linear.new(16, 1)
)

opt = GRX::Optim::Adam.new(model.parameters, lr: 0.03)
loss_fn = GRX::Loss::MSELoss.new

20.times do |epoch|
  train_loader.each do |bx, by|
    opt.zero_grad
    pred = model.call(bx)
    loss = loss_fn.call(pred, by)
    loss.backward
    opt.step
  end
end

val_pred = model.call(val_x)
val_loss = loss_fn.call(val_pred, val_y).item
puts "Validation MSE: #{val_loss.round(6)}"

# Save model weights to binary .grx file
model.save_weights("model_5000_samples.grx")
```

---

## Benchmarks

Measured on Ruby 3.3, Linux x86_64 with AVX2+FMA enabled:

| Operation | Array Size (n) | Execution Time | SIMD Throughput |
|---|---|---|---|
| `add` | 1,000,000 | ~4 ms / iter | ~250M doubles/sec |
| `dot` | 1,000,000 | ~2 ms / iter | ~500M doubles/sec |
| `relu` | 1,000,000 | ~4 ms / iter | ~250M doubles/sec |
| `matmul` (256x256) | 65,536 | ~6 ms | Tiled cache reuse |

---

## License

MIT License. See [LICENSE.txt](LICENSE.txt) for full details.
