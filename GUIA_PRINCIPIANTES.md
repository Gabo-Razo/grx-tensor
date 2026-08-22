# Guia de GRX-Tensor para Principiantes (Desde Cero a Produccion)

**Aprende computacion tensorial, autograd, redes neuronales, chatbots con memoria, NLP, datasets masivos y despliegue en sistemas reales paso a paso.**

---

## Tabla de Contenido

1. [Bienvenida y Filosofia del Framework](#1-bienvenida-y-filosofia-del-framework)
2. [Fundamentos: Que es un Shape, Numel, Strides y Rango?](#2-fundamentos-que-es-un-shape-numel-strides-y-rango)
   - [De Escalar a Tensor Multidimensional](#de-escalar-a-tensor-multidimensional)
   - [Como se guardan los datos en memoria real (C vs Ruby)](#como-se-guardan-los-datos-en-memoria-real-c-vs-ruby)
   - [La formula matematica del indice plano](#la-formula-matematica-del-indice-plano)
3. [Margen de Error y Precision Numerica en GRX](#3-margen-de-error-y-precision-numerica-en-grx)
   - [Doble precision IEEE 754 (Float 64-bit)](#doble-precision-ieee-754-float-64-bit)
   - [Que es FMA (Fused Multiply-Add) y por que reduce el error?](#que-es-fma-fused-multiply-add-y-por-que-reduce-el-error)
   - [Tolerancias (Epsilon) y medicion real del error](#tolerancias-epsilon-y-medicion-real-del-error)
4. [Procesamiento de Lenguaje Natural (NLP): Tokenizacion y Chatbots](#4-procesamiento-de-lenguaje-natural-nlp-tokenizacion-y-chatbots)
   - [Que es la Tokenizacion y como se hace?](#que-es-la-tokenizacion-y-como-se-hace)
   - [De Tokens a Vectores Semanticos (Embedding)](#de-tokens-a-vectores-semanticos-embedding)
   - [Como crear un modelo que hable contigo y use su memoria?](#como-crear-un-modelo-que-hable-contigo-y-use-su-memoria)
   - [Control de Confianza: Que pasa cuando el bot no entiende?](#control-de-confianza-que-pasa-cuando-el-bot-no-entiende)
5. [Como Usar Modelos Entrenados en Produccion (Sin Reentrenar)](#5-como-usar-modelos-entrenados-en-produccion-sin-reentrenar)
   - [El Flujo de Vida: Entrenamiento vs Inferencia](#el-flujo-de-vida-entrenamiento-vs-inferencia)
   - [Ejemplo de API Web con Sinatra / Rails en Finanzas](#ejemplo-de-api-web-con-sinatra--rails-en-finanzas)
6. [Primeros Pasos: Creando Tensores](#6-primeros-pasos-creando-tensores)
7. [Aritmetica y Operaciones Basicas](#7-aritmetica-y-operaciones-basicas)
8. [Algebra Lineal Intuitiva: Multiplicacion de Matrices](#8-algebra-lineal-intuitiva-multiplicacion-de-matrices)
9. [El Superpoder de GRX: Que es Autograd?](#9-el-superpoder-de-grx-que-es-autograd)
10. [Construyendo Redes Neuronales Bloque a Bloque](#10-construyendo-redes-neuronales-bloque-a-bloque)
11. [El Ciclo Sagrado del Entrenamiento (Con loss.backward directo)](#11-el-ciclo-sagrado-del-entrenamiento-con-lossbackward-directo)
12. [Proyectos Guiados Paso a Paso (Completos y Ejecutables)](#12-proyectos-guiados-paso-a-paso-completos-y-ejecutables)
    - [Proyecto 1: El Predictor de Formulas (Regresion Lineal)](#proyecto-1-el-predictor-de-formulas-regresion-lineal)
    - [Proyecto 2: Ajuste de Curvas No Lineales y Guardado en .grx](#proyecto-2-ajuste-de-curvas-no-lineales-y-guardado-en-grx)
    - [Proyecto 3: Clasificador Binario Inteligente con BCELoss](#proyecto-3-clasificador-binario-inteligente-con-bceloss)
    - [Proyecto 4: Entrenamiento con Datasets Masivos (5,000 Filas con DataLoader)](#proyecto-4-entrenamiento-con-datasets-masivos-5000-filas-con-dataloader)
    - [Proyecto 5: Chatbot Financiero Inteligente con Base de Conocimiento y Filtro de Incertidumbre](#proyecto-5-chatbot-financiero-inteligente-con-base-de-conocimiento-y-filtro-de-incertidumbre)
13. [Buenas Practicas y Errores Comunes](#13-buenas-practicas-y-errores-comunes)
14. [Glosario de Terminos Clave](#14-glosario-de-terminos-clave)

---

## 1. Bienvenida y Filosofia del Framework

Ruby es un lenguaje celebrado por su elegancia y legibilidad. Sin embargo, en el procesamiento de matrices masivas e inteligencia artificial, la sobrecarga del recolector de basura (GC) puede restar velocidad si todo se hace en Ruby puro.

Por otro lado, C ofrece velocidad extrema, pero programar grafos de autograd y redes neuronales en C directamente es propenso a errores de memoria.

**GRX-Tensor une lo mejor de ambos mundos:**
- **Ruby habla:** Escribes codigo fluido, declarativo y limpio para definir arquitecturas, procesar datos y crear aplicaciones.
- **C calcula:** Cada operacion matematica, producto matricial, retropropagacion y actualizacion de pesos se ejecuta en un buffer nativo de C con aceleracion vectorial **SIMD (AVX2 + FMA)**, procesando 4 numeros de doble precision por ciclo de CPU.

---

## 2. Fundamentos: Que es un Shape, Numel, Strides y Rango?

### De Escalar a Tensor Multidimensional

Matematicamente, un tensor es una estructura de datos que generaliza escalares, vectores y matrices a cualquier numero de dimensiones:

```
Rango 0 (Escalar):
  Un solo valor numerico.
  Shape: [1]
  Ejemplo: 42.0

Rango 1 (Vector):
  Una secuencia unidimensional de longitud N.
  Shape: [3]
  Visual: [ 1.0, 2.0, 3.0 ]

Rango 2 (Matriz):
  Una tabla bidimensional de Filas x Columnas.
  Shape: [2, 3] (2 filas, 3 columnas)
  Visual:
    ┌                  ┐
    │  1.0   2.0   3.0 │  <- Fila 0
    │  4.0   5.0   6.0 │  <- Fila 1
    └                  ┘

Rango 3 (Tensor 3D):
  Un bloque tridimensional (Profundidad x Filas x Columnas).
  Shape: [2, 2, 3] (2 matrices de 2x3)

Rango 4 (Tensor 4D):
  Comun en procesamiento de imagenes y batches: [Batch, Canales, Alto, Ancho].
  Shape: [64, 3, 224, 224] (64 imagenes RGB de 224x224 pixeles)
```

---

### Conceptos Clave

#### 1. Shape (Forma dimensional)
Es el arreglo de enteros que describe el tamano de cada dimension del tensor.
- Ejemplo: `[5000, 4]` representa una tabla de 5,000 filas y 4 columnas de caracteristicas.

#### 2. Numel (Numero total de elementos)
Es la cantidad total de valores escalares que contiene el tensor:

$$\text{numel} = \prod_{i=0}^{d-1} \text{shape}[i]$$

En Ruby:
```ruby
t = GRX.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
puts t.numel  # => 6 (porque 2 * 3 = 6)
```

#### 3. Strides (Saltos de memoria)
Indica cuantos elementos de la memoria plana contigua se deben saltar para avanzar una posicion en cada eje:
- Para `shape = [2, 3]`, los strides contiguos son `[3, 1]`:
  - Para avanzar 1 columna a la derecha: saltas **1** posicion en memoria.
  - Para avanzar 1 fila hacia abajo: saltas **3** posiciones en memoria.

---

### Como se guardan los datos en memoria real (C vs Ruby)

En la memoria fisica (RAM), los datos existen en una sola linea recta de bytes contiguos:

```
Buffer plano en memoria C:
Posicion C:     [0]    [1]    [2]    [3]    [4]    [5]
Valor:          1.0    2.0    3.0    4.0    5.0    6.0

Interpretacion logica 2D (Shape [2, 3]):
Fila 0:         1.0    2.0    3.0
Fila 1:         4.0    5.0    6.0
```

### La formula matematica del indice plano

$$\text{Indice Plano} = \text{offset} + (\text{fila} \times \text{stride}_0) + (\text{columna} \times \text{stride}_1)$$

Para acceder a `(fila: 1, columna: 1)` con `strides = [3, 1]`:
$$\text{Indice Plano} = 0 + (1 \times 3) + (1 \times 1) = 4 \implies \text{Valor } 5.0$$

---

## 3. Margen de Error y Precision Numerica en GRX

### Doble precision IEEE 754 (Float 64-bit)

A diferencia de librerias que usan 32 bits (`float32`), GRX utiliza **doble precision de 64 bits (`double`)** en todo su nucleo de C:

| Formato | Bits Totales | Bits de Mantisa | Digitos Decimales Significativos | Margen de Error por Redondeo |
|---|---|---|---|---|
| `float32` | 32 bits | 24 bits | ~7 digitos | $\approx 1 \times 10^{-7}$ |
| `double` (GRX) | 64 bits | 53 bits | **~15 a 17 digitos** | $\mathbf{\approx 2 \times 10^{-16}}$ |

### FMA (Fused Multiply-Add)
En el producto punto (`dot`), multiplicacion de matrices (`matmul`) y optimizador `Adam`, se calcula $a \times b + c$:
- **Sin FMA:** Se multiplica $a \times b$, se redondea el resultado, se suma $c$ y se vuelve a redondear.
- **Con FMA en GRX:** La CPU calcula $a \times b + c$ con precision infinita intermedia y hace **un solo redondeo final**, reduciendo a la mitad el margen de error acumulado.

---

## 4. Procesamiento de Lenguaje Natural (NLP): Tokenizacion y Chatbots

Una computadora no puede multiplicar palabras directamente como `"prestamo"` o `"saldo"`. Para que una red neuronal pueda procesar texto, comprender intenciones y responder con lenguaje natural, se sigue un pipeline matematico estricto:

### Que es la Tokenizacion y como se hace?

La tokenizacion es el proceso de convertir un texto en una secuencia ordenada de numeros enteros (*tokens*):

```
Frase del usuario:
"cual es mi saldo disponible hoy"
        │
        ▼
1. Limpieza y separacion en palabras (split):
["cual", "es", "mi", "saldo", "disponible", "hoy"]
        │
        ▼
2. Diccionario de Vocabulario (mapeo de palabra a ID entero):
{ "cual" => 1, "es" => 2, "mi" => 3, "saldo" => 4, "disponible" => 5, "hoy" => 6 }
        │
        ▼
3. Arreglo de IDs de entrada:
[1, 2, 3, 4, 5, 6]
```

En Ruby:
```ruby
vocab = ["cual", "es", "mi", "saldo", "disponible", "hoy"]
token_to_id = vocab.each_with_index.to_h

mensaje = "mi saldo disponible"
ids = mensaje.split.map { |w| token_to_id[w] || 0 }
# => [3, 4, 5]
```

---

### De Tokens a Vectores Semanticos (`Embedding`)

Un simple numero entero como `4` no contiene significado de lo que es un `"saldo"`. Por ello, la capa `GRX::NN::Embedding` traduce cada ID a un vector denso de punto flotante en un espacio continuo de $D$ dimensiones (ej. 16 dimensiones):

```
Token 4 ("saldo") ───> [ 0.42, -0.18, 0.95, ..., -0.04 ] (16 floats)
Token 5 ("dinero") ──> [ 0.40, -0.15, 0.91, ..., -0.02 ] (Muy similar!)
```

Durante el entrenamiento, palabras con significados similares terminan con vectores muy cercanos en el espacio vectorial.

---

### Como crear un modelo que hable contigo y use su memoria?

Para crear un agente conversacional que entienda lo que le pides y recupere informacion de su base de conocimiento:

1. **Base de Conocimiento (Memoria de Hechos):** Se define un diccionario con las respuestas correctas para cada intencion de negocio (saldo, tasas, prestamos, horarios).
2. **Clasificador de Intenciones Neuronal:** La red procesa la frase mediante `Embedding -> LayerNorm -> Linear -> ReLU -> Linear -> Softmax` y calcula probabilidades para cada intencion.
3. **Recuperacion de Memoria:** Al identificar la intencion con mayor probabilidad, el bot extrae la respuesta precisa de su memoria.

---

### Control de Confianza: Que pasa cuando el bot no entiende?

Si un usuario escribe algo fuera de dominio (ej. *"como hago una pizza"*):
- **Verificacion de Cobertura de Vocabulario:** Si la mayoria de las palabras son desconocidas para el modelo ($< 50\%$ de cobertura), se rechaza inmediatamente.
- **Filtro de Probabilidad Softmax (*Confidence Threshold*):** Si la certeza calculada por la red es menor al $70\%$, el bot responde de forma segura:
  > *"Disculpa, no comprendo tu mensaje con suficiente certeza. Solo puedo responder sobre saldo, prestamos o tasas de interes."*

---

## 5. Como Usar Modelos Entrenados en Produccion (Sin Reentrenar)

Una duda muy comun es: *Si entreno una red para finanzas o clasificacion, como la uso en un sistema real (API web, Rails, Sinatra o app de escritorio) sin tener que reentrenarla cada vez?*

### El Flujo de Vida: Entrenamiento vs Inferencia

```
┌─────────────────────────────────────────────────────────────┐
│             FASE 1: ENTRENAMIENTO (Se hace 1 sola vez)      │
│  1. Lees tu dataset historico (5,000+ filas).               │
│  2. Entrenas la red con DataLoader y Adam.                  │
│  3. Guardas los pesos: modelo.save_weights("finanzas.grx")  │
│  4. Guardas el vocabulario / normalizacion en un JSON.      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                Genera el archivo "finanzas.grx"
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             FASE 2: PRODUCCION / INFERENCIA (En tu API)     │
│  1. Al arrancar el servidor web (Rails/Sinatra):            │
│     - Instancias la arquitectura del modelo.                │
│     - Cargas los pesos: modelo.load_weights("finanzas.grx") │
│     - Pones la red en modo inferencia: modelo.eval!         │
│  2. Al recibir una peticion HTTP POST:                      │
│     - Conviertes el JSON de entrada a un Tensor.            │
│     - Ejecutas pred = modelo.call(tensor_entrada).          │
│     - Retornas la respuesta en < 0.1 milisegundos!          │
└─────────────────────────────────────────────────────────────┘
```

### Ejemplo de API Web con Sinatra / Rails en Finanzas

```ruby
# app_api.rb — Microservicio de evaluacion de credito en tiempo real
require "grx"
require "json"

# 1. Cargar el modelo en memoria al iniciar la aplicacion (Toma < 1 ms)
MODELO_RIESGO = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(4, 16),
  GRX::NN::LayerNorm.new(16),
  GRX::NN::Tanh.new,
  GRX::NN::Linear.new(16, 1),
  GRX::NN::Sigmoid.new
)
MODELO_RIESGO.load_weights("modelo_riesgo_crediticio.grx")
MODELO_RIESGO.eval!

# 2. Endpoint de inferencia para evaluar solicitudes de credito
def evaluar_solicitud(ingresos_mensuales, deuda_actual, score_buro, monto_solicitado)
  # Normalizar datos (utilizando las medias y desviaciones del entrenamiento)
  x1 = (ingresos_mensuales - 25000.0) / 10000.0
  x2 = (deuda_actual - 5000.0) / 4000.0
  x3 = (score_buro - 650.0) / 100.0
  x4 = (monto_solicitado - 50000.0) / 30000.0

  entrada = GRX.tensor([x1, x2, x3, x4], [1, 4])
  probabilidad_aprobacion = MODELO_RIESGO.call(entrada).to_a[0]

  decision = probabilidad_aprobacion >= 0.70 ? "APROBADO" : "RECHAZADO"

  {
    decision: decision,
    confianza: (probabilidad_aprobacion * 100).round(2),
    tasa_sugerida: decision == "APROBADO" ? "12.5% anual" : "N/A"
  }
end

# Prueba en produccion:
# res = evaluar_solicitud(45000.0, 2000.0, 720.0, 80000.0)
# puts res.inspect
```

---

## 6. Primeros Pasos: Creando Tensores

```ruby
require "grx"

# 1. Vector de 3 elementos
v = GRX.tensor([1.0, 2.0, 3.0], [3])
puts "Shape: #{v.shape}"  # => [3]
puts "Numel: #{v.numel}"  # => 3

# 2. Matriz de 2 filas x 3 columnas
m = GRX.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
puts "Matriz: #{m.shape}" # => [2, 3]

# 3. Factorias
ceros = GRX.zeros([2, 2]) # Matriz 2x2 de 0.0
unos  = GRX.ones([3, 1])  # Matriz 3x1 de 1.0

# 4. Obtener un valor escalar
scalar = GRX.tensor([99.0], [1])
puts scalar.item  # => 99.0 (Float)
```

---

## 7. Aritmetica y Operaciones Basicas

```ruby
require "grx"

a = GRX.tensor([10.0, 20.0, 30.0], [3])
b = GRX.tensor([2.0,  5.0,  10.0], [3])

# Operaciones tensor-tensor (C + SIMD)
puts (a + b).to_a  # => [12.0, 25.0, 40.0]
puts (a * b).to_a  # => [20.0, 100.0, 300.0]

# Operaciones con escalares
puts (a + 5.0).to_a   # => [15.0, 25.0, 35.0]
puts (2.0 * a).to_a   # => [20.0, 40.0, 60.0]

# Reducciones diferenciables
x = GRX.tensor([1.0, 4.0, 9.0, 16.0], [4])
puts x.sum.item       # => 30.0
puts x.mean.item      # => 7.5
```

---

## 8. Algebra Lineal Intuitiva: Multiplicacion de Matrices

```ruby
require "grx"

# [2, 3] x [3, 2] -> [2, 2]
a = GRX.tensor([1.0, 2.0, 3.0,
                4.0, 5.0, 6.0], [2, 3])

b = GRX.tensor([7.0,  8.0,
                9.0,  10.0,
                11.0, 12.0], [3, 2])

c = a.matmul(b)
puts c.shape  # => [2, 2]
puts c.to_a   # => [58.0, 64.0, 139.0, 154.0]
```

---

## 9. El Superpoder de GRX: Que es Autograd?

```ruby
require "grx"

x = GRX.tensor([2.0, 3.0], [2], requires_grad: true)
y = GRX.tensor([4.0, 5.0], [2], requires_grad: true)

# Computamos z = (x + y) * y
z = ((x + y) * y).sum
z.backward

# Derivadas exactas calculadas automaticamente
puts x.grad.to_a  # => [4.0, 5.0]
puts y.grad.to_a  # => [10.0, 13.0]
```

---

## 10. Construyendo Redes Neuronales Bloque a Bloque

```ruby
require "grx"

modelo = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(4, 16),
  GRX::NN::LayerNorm.new(16),
  GRX::NN::ReLU.new,
  GRX::NN::Linear.new(16, 1)
)

puts modelo
```

---

## 11. El Ciclo Sagrado del Entrenamiento (Con `loss.backward` directo)

```ruby
opt.zero_grad                         # 1. Limpiar gradientes anteriores
pred = modelo.call(train_x)           # 2. Forward pass
loss = loss_fn.call(pred, train_y)    # 3. Calcular la perdida
loss.backward                         # 4. Backward pass automatico
opt.step                              # 5. Ajustar pesos
```

---

## 12. Proyectos Guiados Paso a Paso (Completos y Ejecutables)

---

### Proyecto 1: El Predictor de Formulas (Regresion Lineal)

Aprende la relacion $y = 3x + 2$ usando `loss.backward` nativo.

```ruby
require "grx"

train_x = GRX.tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5, 1])
train_y = GRX.tensor([5.0, 8.0, 11.0, 14.0, 17.0], [5, 1])

modelo = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(1, 1)
)

opt = GRX::Optim::Adam.new(modelo.parameters, lr: 0.1)
loss_fn = GRX::Loss::MSELoss.new

150.times do
  opt.zero_grad
  pred = modelo.call(train_x)
  loss = loss_fn.call(pred, train_y)
  loss.backward
  opt.step
end

test_x = GRX.tensor([6.0], [1, 1])
puts "Prediccion para x=6 (esperado=20): #{modelo.call(test_x).to_a[0].round(3)}"
```

---

### Proyecto 2: Ajuste de Curvas No Lineales y Guardado en `.grx`

```ruby
require "grx"

train_x = GRX.tensor((1..10).map(&:to_f), [10, 1])
train_y = GRX.tensor((1..10).map { |x| 2.0 * x + 1.0 }, [10, 1])

mlp = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(1, 16),
  GRX::NN::LayerNorm.new(16),
  GRX::NN::Tanh.new,
  GRX::NN::Linear.new(16, 1)
)

opt = GRX::Optim::Adam.new(mlp.parameters, lr: 0.05)
loss_fn = GRX::Loss::MSELoss.new

300.times do
  opt.zero_grad
  pred = mlp.call(train_x)
  loss = loss_fn.call(pred, train_y)
  loss.backward
  GRX::Utils.clip_grad_norm!(mlp.parameters, 1.0)
  opt.step
end

# Guardar pesos a disco
mlp.save_weights("/tmp/mlp_entrenado.grx")

# Cargar en una nueva instancia limpia para inferencia
modelo_nuevo = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(1, 16),
  GRX::NN::LayerNorm.new(16),
  GRX::NN::Tanh.new,
  GRX::NN::Linear.new(16, 1)
)
modelo_nuevo.load_weights("/tmp/mlp_entrenado.grx")

puts "Inferencia x=11: #{modelo_nuevo.call(GRX.tensor([11.0], [1, 1])).to_a[0].round(3)} (Esperado: 23.0)"
```

---

### Proyecto 3: Clasificador Binario Inteligente con BCELoss

```ruby
require "grx"

train_x = GRX.tensor([0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  1.0, 1.0], [4, 2])
train_y = GRX.tensor([0.0,        1.0,        1.0,        1.0],       [4, 1])

clasificador = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(2, 4),
  GRX::NN::Tanh.new,
  GRX::NN::Linear.new(4, 1),
  GRX::NN::Sigmoid.new
)

opt = GRX::Optim::Adam.new(clasificador.parameters, lr: 0.08)
loss_fn = GRX::Loss::BCELoss.new

400.times do
  opt.zero_grad
  pred = clasificador.call(train_x)
  loss = loss_fn.call(pred, train_y)
  loss.backward
  opt.step
end

puts "Resultados OR: #{clasificador.call(train_x).to_a.map { |v| v.round(4) }}"
```

---

### Proyecto 4: Entrenamiento con Datasets Masivos (5,000 Filas con DataLoader)

```ruby
require "grx"

num_samples = 5000
num_features = 4

# y = 2*x1 - 3*x2 + 1.5*x3 - 0.5*x4 + 4.0
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

val_loss = loss_fn.call(model.call(val_x), val_y).item
puts "Error de Validacion MSE (5,000 muestras): #{val_loss.round(6)}"
```

---

### Proyecto 5: Chatbot Financiero Inteligente con Base de Conocimiento y Filtro de Incertidumbre

Este proyecto crea un agente de atencion al cliente que comprende intenciones mediante redes neuronales y recupera respuestas exactas de su memoria. Si una consulta es incomprensible o fuera de tema, lo reconoce honestamente:

```ruby
require "grx"

# 1. Base de conocimiento (Memoria del asistente)
KNOWLEDGE_BASE = {
  "saludo" => [
    "Hola, soy tu asistente financiero. En que puedo ayudarte hoy?",
    "Que tal! Estoy listo para responder tus dudas financieras y de cuenta."
  ],
  "consultar_saldo" => [
    "Tu saldo disponible actual es de $14,850.50 MXN en tu cuenta principal.",
    "Actualmente cuentas con $14,850.50 MXN listos para operar."
  ],
  "tasa_interes" => [
    "Nuestra tasa de rendimiento anual fija actual es del 11.5% anual.",
    "La tasa de interes en cuentas de inversion es de 11.5% anual con pago mensual."
  ],
  "prestamo" => [
    "Puedes solicitar un prestamo de hasta $100,000 MXN directamente desde la app.",
    "Para prestamos personales, ofrecemos plazos de 6 a 36 meses con aprobacion inmediata."
  ],
  "despedida" => [
    "Hasta luego! Que tengas un excelente dia.",
    "Nos vemos, vuelve pronto si necesitas mas informacion."
  ]
}

# 2. Frases de entrenamiento
training_phrases = [
  ["hola buenos dias", "saludo"],
  ["hola que tal", "saludo"],
  ["buenas tardes", "saludo"],
  ["cuanto dinero tengo en mi cuenta", "consultar_saldo"],
  ["cual es mi saldo disponible hoy", "consultar_saldo"],
  ["cuanto saldo me queda", "consultar_saldo"],
  ["que tasa de interes tienen", "tasa_interes"],
  ["cuanto rendimiento me da la inversion", "tasa_interes"],
  ["como puedo pedir un prestamo", "prestamo"],
  ["necesito un credito personal", "prestamo"],
  ["quiero solicitar un prestamo", "prestamo"],
  ["adios muchas gracias por todo", "despedida"],
  ["hasta luego nos vemos pronto", "despedida"]
]

# 3. Tokenizacion y vocabulario
intents = KNOWLEDGE_BASE.keys
intent_to_id = intents.each_with_index.to_h
id_to_intent = intent_to_id.invert
num_classes  = intents.size

vocab = training_phrases.flat_map { |text, _| text.split }.uniq
token_to_id = vocab.each_with_index.to_h
vocab_size  = vocab.size
embedding_dim = 16
seq_len = 6

batch_size = training_phrases.size
x_ids = training_phrases.map do |text, _|
  tokens = text.split.map { |w| token_to_id[w] }
  (tokens + [0] * seq_len).first(seq_len)
end
train_x = GRX.tensor(x_ids.flatten.map(&:to_f), [batch_size, seq_len])

onehot = Array.new(batch_size * num_classes, 0.0)
training_phrases.each_with_index { |(_, label), i| onehot[i * num_classes + intent_to_id[label]] = 1.0 }
train_y = GRX.tensor(onehot, [batch_size, num_classes])

# 4. Arquitectura de la Red
emb = GRX::NN::Embedding.new(vocab_size, embedding_dim)
ln  = GRX::NN::LayerNorm.new(embedding_dim)
fc1 = GRX::NN::Linear.new(embedding_dim, 16)
act = GRX::NN::ReLU.new
fc2 = GRX::NN::Linear.new(16, num_classes)

params = emb.parameters + ln.parameters + fc1.parameters + fc2.parameters
opt = GRX::Optim::Adam.new(params, lr: 0.05)
loss_fn = GRX::Loss::CrossEntropyLoss.new

# 5. Entrenamiento
150.times do
  opt.zero_grad
  flat = train_x.flatten
  e = emb.call(flat)
  e_data = e.to_a
  pooled_data = Array.new(batch_size * embedding_dim, 0.0)
  batch_size.times do |b|
    embedding_dim.times do |d|
      s = 0.0
      seq_len.times { |t| s += e_data[(b * seq_len + t) * embedding_dim + d] }
      pooled_data[b * embedding_dim + d] = s / seq_len.to_f
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
        v = p_grad[b * embedding_dim + d] / seq_len.to_f
        seq_len.times { |t| grad_emb[(b * seq_len + t) * embedding_dim + d] = v }
      end
    end
    e.backward(GRX.tensor(grad_emb, e.shape))
  end
  opt.step
end

# 6. Motor de conversacion con control de confianza
def chatear(mensaje, emb, ln, fc1, act, fc2, token_to_id, id_to_intent, seq_len, embedding_dim)
  words = mensaje.downcase.gsub(/[^a-z0-9\s]/, "").split
  known_count = words.count { |w| token_to_id.key?(w) }
  coverage = words.empty? ? 0.0 : known_count.to_f / words.size

  # Si la mayoria de las palabras son desconocidas, rechazar amablemente
  if coverage < 0.40
    return "Bot: Disculpa, no comprendo tu mensaje. Solo puedo responder sobre saldo, prestamos o tasas."
  end

  tokens = words.map { |w| token_to_id[w] || 0 }
  padded = (tokens + [0] * seq_len).first(seq_len)
  t_in = GRX.tensor(padded.map(&:to_f), [seq_len])
  e = emb.call(t_in)
  e_data = e.to_a

  m_data = Array.new(embedding_dim) do |d|
    seq_len.times.sum { |t| e_data[t * embedding_dim + d] } / seq_len.to_f
  end
  p_t = GRX.tensor(m_data, [1, embedding_dim])
  logits = fc2.call(act.call(fc1.call(ln.call(p_t))))
  probs = logits.softmax.to_a

  best_prob, best_idx = probs.each_with_index.max_by { |p, idx| p }

  if best_prob < 0.70
    "Bot: No estoy seguro de haberte entendido (#{(best_prob * 100).round(1)}% certeza). Podrias reformular?"
  else
    intent = id_to_intent[best_idx]
    respuesta = KNOWLEDGE_BASE[intent].sample
    "Bot: #{respuesta} (Certeza: #{(best_prob * 100).round(1)}%)"
  end
end

# 7. Prueba interactiva
puts "\n--- Simulacion de Chat ---"
[
  "Hola muy buenos dias",
  "Oye cuanto dinero tengo en mi cuenta",
  "Que tasa de interes tienen actualmente",
  "Quiero pedir un prestamo personal",
  "Como puedo cocinar una hamburguesa",  # Fuera de tema
  "Adios muchas gracias"
].each do |msg|
  puts "\nUsuario: #{msg}"
  puts chatear(msg, emb, ln, fc1, act, fc2, token_to_id, id_to_intent, seq_len, embedding_dim)
end
```

---

## 13. Buenas Practicas y Errores Comunes

1. **Llamar `opt.zero_grad` en cada iteracion:** Evita que los gradientes se acumulen indefinidamente.
2. **Dimensiones compatibles:** Las capas lineales esperan siempre tensores 2D `[batch_size, num_features]`.
3. **Manejar `train!` y `eval!`:** Modos necesarios para capas como `Dropout` y `BatchNorm1d` durante inferencia.

---

## 14. Glosario de Terminos Clave

- **Shape:** Dimensiones del tensor (ej. `[10, 4]` -> 10 filas, 4 columnas).
- **Numel:** Total de elementos del tensor.
- **Strides:** Saltos de memoria plana para avanzar en cada eje.
- **Tokenizacion:** Division y conversion de texto a identificadores enteros.
- **Embedding:** Capa que transforma tokens en vectores densos continuos.
- **CrossEntropyLoss:** Perdida multiclase para clasificar texto e intenciones.
- **FMA:** Fused Multiply-Add en CPU que calcula $a \times b + c$ en 1 solo ciclo con minima desviacion.
