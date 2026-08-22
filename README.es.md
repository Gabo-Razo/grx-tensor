# GRX-Tensor

**Ruby habla. C calcula.**

Un framework de tensores para Ruby con diferenciacion automatica (autograd), un nucleo de computo en C con instrucciones vectoriales SIMD, y primitivas completas para redes neuronales, todo respaldado por una API de Ruby limpia y expresiva.

[![Ruby](https://img.shields.io/badge/ruby-%3E%3D%203.0-CC342D?logo=ruby)](https://www.ruby-lang.org)
[![Licencia: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE.txt)
[![Plataforma](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/Gabo-Razo/grx-tensor)

---

## Que es GRX?

GRX es una biblioteca de computacion tensorial de alto rendimiento para Ruby. Su nucleo numerico esta implementado en C y compilado con extensiones **AVX2 + FMA** SIMD — procesando 4 numeros de punto flotante de doble precision (doubles) por ciclo de reloj con multiplicacion-suma fusionada.

Ruby gestiona la interfaz de usuario de alto nivel: validacion de formas dimensionales (shapes), construccion del grafo de computacion y orquestacion. C maneja la memoria nativa y las operaciones aritmeticas pesadas.

### Caracteristicas clave

| Caracteristica | Detalle |
|---|---|
| **Nucleo C+SIMD** | AVX2+FMA, soporte SSE2 y fallback escalar autodetectado al compilar |
| **Memoria Alineada** | Asignacion en el heap de C alineada a 32 bytes (`posix_memalign` / `_aligned_malloc`) |
| **Autograd** | Diferenciacion automatica con funciones de perdida 100% diferenciables |
| **Optimizadores** | SGD (con momento y weight decay) y Adam (vectorizado en C con FMA) |
| **Capas de Redes Neuronales** | Linear, Sequential, Embedding, LayerNorm, Dropout, BatchNorm1d |
| **Funciones de Activacion** | ReLU, Leaky ReLU, Tanh, Sigmoid, Softmax (todas diferenciables) |
| **Funciones de Perdida** | MSE, MAE, BCE, CrossEntropy, Huber (totalmente diferenciables) |
| **Persistencia de Modelos** | Formato binario nativo de alta velocidad `.grx` (`save_weights` / `load_weights`) |
| **Pipelines de Datos** | `TensorDataset` y `DataLoader` para procesamiento por lotes (*batching*) y barajado (*shuffle*) |
| **Inicializacion de Pesos** | Xavier uniforme, He normal (Box-Muller en C) |
| **Multiplataforma** | Linux (`.so`), macOS (`.dylib`), Windows (`.dll`) |
| **Fallback Ruby Puro** | Funciona sin compilador de C en caso de que los binarios no esten disponibles |

---

## Instalacion

### Mediante RubyGems

```bash
gem install grx-tensor
```

O agregalo a tu `Gemfile`:

```ruby
gem "grx-tensor"
```

### Compilacion Manual desde el Codigo Fuente

Si clonas el repositorio directamente, compila la extension nativa:

**Linux / macOS:**
```bash
make -C ext/unix
```

**Windows (MinGW-w64):**
```bash
make -C ext/windows -f Makefile.mingw
```

---

## Inicio Rapido

```ruby
require "grx"

# 1. Crear tensores con seguimiento de gradientes
a = GRX.tensor([1.0, 2.0, 3.0], [3], requires_grad: true)
b = GRX.tensor([4.0, 5.0, 6.0], [3], requires_grad: true)

# 2. Realizar operaciones aritmeticas (ejecutadas en C con SIMD)
c = (a * b) + 2.0

# 3. Calcular gradientes mediante retropropagacion (backpropagation)
c.sum.backward

# 4. Inspeccionar los gradientes calculados
puts a.grad.to_a  # [4.0, 5.0, 6.0]
puts b.grad.to_a  # [1.0, 2.0, 3.0]
```

---

## Arquitecturas Avanzadas: NLP y Tablas de Incrustacion

```ruby
require "grx"

# 1. Vocabulario de 1000 palabras mapeadas a vectores de 32 dimensiones
vocabulario_size = 1000
embedding_dim = 32
emb = GRX::NN::Embedding.new(vocabulario_size, embedding_dim)

# 2. Tokens de entrada
tokens = GRX.tensor([42, 108, 999], [3])

# 3. Obtener vectores densos
vectores = emb.call(tokens)
puts vectores.shape  # [3, 32]

# 4. Normalizacion por capa (LayerNorm)
ln = GRX::NN::LayerNorm.new(32)
normalizado = ln.call(vectores)
puts normalizado.shape  # [3, 32]
```

---

## Ejemplos Completos y Funcionales

### Ejemplo 1: Clasificador de Texto y Sentimiento con NLP (CrossEntropyLoss)

Clasifica consultas de usuarios en 3 categorias: Queja (0), Elogio (1) y Consulta (2):

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

puts "Entrenamiento NLP completado con perdida CrossEntropy < 0.0001"
```

---

### Ejemplo 2: Entrenamiento con Datasets Masivos (5,000 Muestras con DataLoader)

```ruby
require "grx"

num_samples = 5000
num_features = 4

# Funcion subyacente: y = 2*x1 - 3*x2 + 1.5*x3 - 0.5*x4 + 4.0
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
puts "Error de Validacion MSE: #{val_loss.round(6)}"

# Guardar pesos a disco en formato binario .grx
model.save_weights("modelo_5000_muestras.grx")
```

---

## Pruebas de Rendimiento (Benchmark)

Medido en Ruby 3.3, Linux x86_64 con extensiones AVX2+FMA activadas:

| Operacion | Tamano (n) | Tiempo por iteracion | Rendimiento SIMD |
|---|---|---|---|
| `add` | 1,000,000 | ~4 ms | ~250M doubles/segundo |
| `dot` | 1,000,000 | ~2 ms | ~500M doubles/segundo |
| `relu` | 1,000,000 | ~4 ms | ~250M doubles/segundo |
| `matmul` (256x256) | 65,536 | ~6 ms | Reutilizacion de cache por bloques (tiling) |

---

## Licencia

Licencia MIT. Consulta el archivo [LICENSE.txt](LICENSE.txt) para mas detalles.
