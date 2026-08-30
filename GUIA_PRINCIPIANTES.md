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
11. [El Ciclo Sagrado del Entrenamiento (Explicacion Paso a Paso)](#11-el-ciclo-sagrado-del-entrenamiento-explicacion-paso-a-paso)
12. [El Diccionario Sagrado de Parametros e Hiperparametros](#12-el-diccionario-sagrado-de-parametros-e-hiperparametros)
13. [Proyectos Guiados Paso a Paso (Completos y Ejecutables)](#13-proyectos-guiados-paso-a-paso-completos-y-ejecutables)
    - [Proyecto 1: El Conversor de Temperatura (Celsius a Fahrenheit)](#proyecto-1-el-conversor-de-temperatura-celsius-a-fahrenheit)
    - [Proyecto 2: Compuertas Logicas (AND Lineal vs XOR No Lineal)](#proyecto-2-compuertas-logicas-and-lineal-vs-xor-no-lineal)
    - [Proyecto 3: El Predictor de Formulas (Regresion Lineal)](#proyecto-3-el-predictor-de-formulas-regresion-lineal)
    - [Proyecto 4: Ajuste de Curvas No Lineales y Guardado en .grx](#proyecto-4-ajuste-de-curvas-no-lineales-y-guardado-en-grx)
    - [Proyecto 5: Clasificador Binario Inteligente con BCELoss](#proyecto-5-clasificador-binario-inteligente-con-bceloss)
    - [Proyecto 6: Entrenamiento con Datasets Masivos (5,000 Filas con DataLoader)](#proyecto-6-entrenamiento-con-datasets-masivos-5000-filas-con-dataloader)
    - [Proyecto 7: Chatbot Financiero Inteligente con Base de Conocimiento y Filtro de Incertidumbre](#proyecto-7-chatbot-financiero-inteligente-con-base-de-conocimiento-y-filtro-de-incertidumbre)
14. [Buenas Practicas y Errores Comunes](#14-buenas-practicas-y-errores-comunes)
15. [Glosario de Terminos Clave](#15-glosario-de-terminos-clave)

---

## 1. Bienvenida y Filosofia del Framework

Ruby es un lenguaje celebrado por su elegancia y legibilidad. Sin embargo, en el procesamiento de matrices masivas e inteligencia artificial, la sobrecarga del recolector de basura (GC) puede restar velocidad si todo se hace en Ruby puro.

Por otro lado, C ofrece velocidad extrema, pero programar grafos de autograd y redes neuronales en C directamente es propenso a errores de memoria.

**GRX-Tensor une lo mejor de ambos mundos:**
- **Ruby habla:** Escribes codigo fluido, declarativo y limpio para definir arquitecturas, procesar datos y crear aplicaciones.
- **C calcula:** Cada operacion matematica, producto matricial, retropropagacion y actualizacion de pesos se ejecuta en un buffer nativo de C con aceleracion vectorial **SIMD multi-target (AVX2 + FMA, SSE y C escalar)**.

### Instalacion y Compatibilidad de Hardware Universal
1. **Instalacion 100% automatica:** Al instalar con `gem install grx-tensor`, RubyGems compila y enlaza automaticamente la extension nativa de C en segundo plano sin requerir comandos de compilacion manuales.
2. **Despacho Dinamico por CPU:** El motor en C detecta en tiempo real las caracteristicas de tu procesador:
   - Si tu procesador soporta **AVX2 + FMA**, activa la maxima aceleracion vectorial (4 doubles/ciclo).
   - Si tu procesador soporta **SSE**, activa las instrucciones vectoriales SSE.
   - Si tu maquina es ARM, una maquina virtual o una CPU clasica, conmuta a **C escalar**, evitando cualquier error de "instruccion ilegal".
3. **Estado del Soporte en Windows:** En Windows, la aceleracion nativa funciona compilando automaticamente mediante **RubyInstaller con DevKit (MSYS2 / MinGW-w64)**. Si no hay compilador presente, el framework corre en modo fallback de Ruby puro de forma segura. El empaquetado de binarios pre-compilados (.dll) sin necesidad de DevKit se encuentra en desarrollo activo.

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

## 11. El Ciclo Sagrado del Entrenamiento (Explicacion Paso a Paso)

El entrenamiento de cualquier modelo en GRX sigue un ciclo de 5 pasos universales:

```ruby
# 1. Limpiar gradientes anteriores
#    En cada paso, el optimizador calcula la direccion del cambio.
#    Si no limpias los gradientes, se acumularan como una bola de nieve.
opt.zero_grad

# 2. Pase hacia adelante (Forward Pass)
#    Los datos entran a la red y se procesan a traves de capas de matrices en C con SIMD.
pred = modelo.call(train_x)

# 3. Calculo de la Perdida (Loss / Error)
#    La funcion de perdida mide numericamente que tan lejos estamos de la realidad.
loss = loss_fn.call(pred, train_y)

# 4. Pase hacia atras (Backward Pass / Autograd)
#    GRX recorre el grafo DAG hacia atras calculando la derivada exacta de cada peso.
loss.backward

# 5. Paso de Optimizacion (Optimizer Step)
#    El optimizador (Adam o SGD) actualiza los pesos en C para reducir el error.
opt.step
```

---

## 12. El Diccionario Sagrado de Parametros e Hiperparametros

Cuando creas optimizadores, capas o funciones de perdida, veras expresiones como `lr: 0.001`, `lr: 1e-3`, `momentum: 0.9`, `weight_decay: 1e-4`, `eps: 1e-8` o `betas: [0.9, 0.999]`. Que significan, que valores admiten y como se deben configurar?

---

### 1. Que significa la Notacion Cientifica en Machine Learning (`1e-1` a `1e-8`)?

En Ruby y en Inteligencia Artificial, los numeros muy pequenos se escriben en **notacion cientifica exponencial** (`1e-X` que equivale a $1.0 \times 10^{-X}$). Esto evita errores tipograficos al contar ceros decimales:

| Notacion Ruby | Valor Decimal Exacto | Fraccion | Nombre Comun | Uso Tipico en GRTensor |
|---|---|---|---|---|
| `1e-1` | `0.1` | $1/10$ | Una decima | Momentum de actualizacion en `BatchNorm1d`, learning rate rapido. |
| `1e-2` | `0.01` | $1/100$ | Una centesima | Learning rate estandar para `SGD`, pendiente `alpha` en `LeakyReLU`. |
| `1e-3` | `0.001` | $1/1,000$ | Una milesima | Learning rate estandar de oro para `Adam` en redes profundas. |
| `1e-4` | `0.0001` | $1/10,000$ | Una diezmilesima | Penalizacion de regularizacion L2 (`weight_decay`) para evitar sobreajuste. |
| `1e-5` | `0.00001` | $1/100,000$ | Una cienmilesima | Epsilon (`eps`) de varianza en `LayerNorm` y `BatchNorm1d`. |
| `1e-7` | `0.0000001` | $1/10,000,000$ | Una diezmillonesima | Epsilon de corte en `BCELoss` para evitar $\log(0) \to -\infty$. |
| `1e-8` | `0.00000001` | $1/100,000,000$ | Una cienmillonesima | Epsilon de estabilidad numerica en el denominador de `Adam`. |

---

### 2. Parametros de Optimizadores (`GRX::Optim`)

#### `lr` (Learning Rate / Tasa de Aprendizaje)
* **Que es:** La velocidad y tamano del paso que da la red hacia el error minimo en cada iteracion.
* **Analogia:** Caminar hacia el fondo de un valle con los ojos vendados.
  * Si `lr` es muy grande (`lr: 10.0`), saltaras de una colina a otra sin tocar el fondo y la perdida explotara (`Infinity` o `NaN`).
  * Si `lr` es muy pequeno (`lr: 1e-6`), tardaras semanas en dar 3 pasos.
* **Valores recomendados:**
  * `0.001` o `1e-3` (Valor por defecto de oro para `Adam` en casi todas las redes neuronales).
  * `0.01` a `0.1` (`1e-2` a `1e-1`) (Para `SGD` con momentum o regresiones lineales rapidas).
  * `0.5` a `0.8` (Para modelos de 1 sola neurona como el conversor de temperatura).

#### `momentum` (Momento / Inercia en SGD)
* **Que es:** Acumula la velocidad y direccion de los gradientes anteriores para acelerar el descenso.
* **Analogia:** Una bola pesada de boliche rodando colina abajo. En lugar de detenerse o rebotar caoticamente por cada bache diminuto en los datos, la bola mantiene su inercia hacia adelante.
* **Valores:** `0.0` (sin momento) a `0.99`. Valor recomendado estandar: `0.9`.

#### `weight_decay` (Decaimiento de Pesos / Regularizacion L2)
* **Que es:** Una penalizacion matematica que encoge ligeramente los pesos hacia cero en cada paso ($w \leftarrow w - \lambda \cdot w$).
* **Analogia:** La Navaja de Ockham. Evita que la red memorice las respuestas exactas (*Overfitting* o Sobreajuste) forzandola a preferir explicaciones simples y pesos pequenos y equilibrados.
* **Valores recomendados:** `0.0` (por defecto, sin penalizacion), `1e-4` ($0.0001$) o `1e-5` para datasets del mundo real.

#### `betas` / `beta1`, `beta2` (En `GRX::Optim::Adam`)
* **Que es:** Las tasas de decaimiento exponencial para la estimacion de momentos de 1er y 2do orden.
  * $\beta_1$ (defecto: `0.9`): Memoria del 90% de la direccion del gradiente anterior (momento direccional).
  * $\beta_2$ (defecto: `0.999`): Memoria del 99.9% de la varianza del gradiente al cuadrado (escala automaticamente pasos grandes para gradientes raros y pasos pequenos para gradientes frecuentes).
* **Valores recomendados:** `[0.9, 0.999]` (el estandar probado de Kingma & Ba).

#### `eps` / `epsilon` (Estabilidad Numerica)
* **Que es:** Un numero microscopico ($1e-8$ en Adam, $1e-5$ en LayerNorm/BatchNorm) sumado al denominador.
* **Por que existe:** Si la varianza de un parametro es 0, dividir entre 0 causaria un fallo critico de hardware (`NaN`). El epsilon actua como cinturon de seguridad matematico.

---

### 3. Parametros de Capas Neuronales (`GRX::NN`)

| Capa / Modulo | Parametro | Tipo | Por Defecto | Rango Valido | Explicacion Didactica |
|---|---|---|---|---|---|
| `GRX::NN::Linear` | `in_features` | Integer | (Requerido) | $\ge 1$ | Cantidad de numeros de entrada que recibe la capa. |
| `GRX::NN::Linear` | `out_features` | Integer | (Requerido) | $\ge 1$ | Cantidad de neuronas o caracteristicas que produce de salida. |
| `GRX::NN::Linear` | `bias` | Boolean | `true` | `true` / `false` | Si es `true`, agrega el termino de sesgo $b$ ($y = Wx + b$). |
| `GRX::NN::Embedding`| `num_embeddings` | Integer | (Requerido) | $\ge 1$ | Tamano del vocabulario o numero total de entidades unicas. |
| `GRX::NN::Embedding`| `embedding_dim` | Integer | (Requerido) | $\ge 1$ | Dimension del vector denso continuo para representar cada palabra. |
| `GRX::NN::Dropout` | `p` | Float | `0.5` | `0.0` a `0.99` | Probabilidad de desactivar aleatoriamente una neurona durante el entrenamiento (0.2 = 20%, 0.5 = 50%) para evitar que la red se vuelva dependiente de una sola neurona. |
| `GRX::NN::LeakyReLU`| `alpha` | Float | `0.01` | `0.001` a `0.3` | Pendiente para numeros negativos. Evita que las neuronas "mueran" permitiendo pasar un 1% de gradiente cuando $x < 0$. |
| `GRX::NN::LayerNorm`| `normalized_shape`| Integer/Array | (Requerido) | Dimensiones | Dimension sobre la que se calcula la media y varianza unitaria. |
| `GRX::NN::LayerNorm`| `eps` / `epsilon` | Float | `1e-5` | `1e-8` a `1e-4` | Termino sumado a la varianza para evitar division entre 0. |
| `GRX::NN::BatchNorm1d`| `num_features` | Integer | (Requerido) | $\ge 1$ | Cantidad de canales a normalizar a lo largo del lote (*batch*). |
| `GRX::NN::BatchNorm1d`| `eps` / `epsilon` | Float | `1e-5` | `1e-8` a `1e-4` | Termino de estabilidad sumado a la varianza por lote. |
| `GRX::NN::BatchNorm1d`| `momentum` | Float | `0.1` | `0.01` a `0.5` | Factor de actualizacion de medias y varianzas moviles para inferencia. |

---

### 4. Parametros de Funciones de Perdida (`GRX::Loss`)

| Funcion de Perdida | Parametro | Opciones / Tipo | Por Defecto | Explicacion |
|---|---|---|---|---|
| `MSELoss` / `MAELoss` | `reduction` | Symbol | `:mean` | `:mean` promedia el error entre todas las muestras del lote. `:sum` suma todos los errores directos. |
| `BCELoss` | `reduction` | Symbol | `:mean` | `:mean` o `:sum`. |
| `BCELoss` | `eps` | Float | `1e-7` | Limite de seguridad para evitar $\log(0) \to -\infty$ en probabilidades extremas (0.0 o 1.0). |
| `CrossEntropyLoss` | `reduction` | Symbol | `:mean` | `:mean` o `:sum`. Aplica Softmax con Log-Sum-Exp. |
| `HuberLoss` | `delta` | Float | `1.0` | Umbral de transicion: si el error es menor a $\delta$ se calcula como cuadratico (MSE); si es mayor, se calcula lineal (MAE) para no volverse loco con datos atipicos (*outliers*). |
| `HuberLoss` | `reduction` | Symbol | `:mean` | `:mean` o `:sum`. |

---

### 5. Parametros de Creacion y Manipulacion de Tensores (`GRX::Tensor`)

| Metodo / Factory | Parametro | Tipo | Por Defecto | Descripcion |
|---|---|---|---|---|
| `GRX.tensor` | `data` | Array / Storage | (Requerido) | Datos numericos en arreglo plano (`[1.0, 2.0]`) o anidado (`[[1, 2], [3, 4]]`). |
| `GRX.tensor` | `shape` | Array[Integer] | `nil` (Auto) | Dimensiones del tensor (ej. `[2, 3]` para 2 filas y 3 columnas). |
| `GRX.tensor` | `requires_grad`| Boolean | `false` | Activa el rastreo en el grafo DAG para calcular gradientes con `backward`. |
| `GRX.zeros` / `GRX.ones` | `shape` | Array[Integer] | (Requerido) | Dimensiones del tensor a inicializar con ceros o unos. |
| `GRX.rand` | `shape` | Array[Integer] | (Requerido) | Genera tensor con distribucion uniforme $U[0, 1)$. |
| `GRX.randn` | `shape` | Array[Integer] | (Requerido) | Genera tensor con distribucion normal estandar $N(0, 1)$ mediante Box-Muller. |
| `Tensor.xavier_uniform`| `shape` | Array[Integer] | (Requerido) | Inicializador Xavier/Glorot ($U[-\sqrt{6/(f_{in}+f_{out})}, \sqrt{6/(f_{in}+f_{out})}]$). |
| `Tensor.he_normal` | `shape` | Array[Integer] | (Requerido) | Inicializador He/Kaiming ($N(0, \sqrt{2/f_{in}})$), optimo para capas con ReLU. |
| `Tensor.zeros_like` | `other` | Tensor | (Requerido) | Crea un tensor de ceros con la misma forma que `other`. |
| `Tensor.ones_like` | `other` | Tensor | (Requerido) | Crea un tensor de unos con la misma forma que `other`. |
| `tensor.clip` | `lo`, `hi` | Numeric | (Requeridos) | Limites inferior y superior. Fija valores fuera del intervalo $[lo, hi]$. |
| `tensor.pow` | `exponent` | Numeric | (Requerido) | Exponente al que se eleva cada elemento ($x^e$). Totalmente diferenciable. |
| `tensor.reshape` | `new_shape` | Array[Integer] | (Requerido) | Nueva forma manteniendo el mismo `numel` total. Vista zero-copy. |
| `tensor.transpose` | (sin args) | - | - | Intercambia filas y columnas en tensores 2D. Vista zero-copy por strides. |
| `tensor.flatten` | (sin args) | - | - | Aplana el tensor a 1 dimension `[numel]`. Vista zero-copy. |
| `tensor.contiguous`| (sin args) | - | - | Re-empaca la memoria no contigua (tras un transpose) en un buffer contiguo nuevo. |
| `tensor.get` | `*indices` | Integers | (Requerido) | Coordenadas del elemento a leer (ej. `t.get(0, 2)`). |
| `tensor.set` | `*indices, val`| Integers, Float | (Requeridos) | Modifica directamente el valor en la posicion especificada. |
| `tensor.item` | (sin args) | - | - | Extrae el valor numerico Float de un tensor escalar de 1 solo elemento. |
| `tensor.argmax` | (sin args) | - | - | Retorna el indice del elemento con el valor maximo. |
| `tensor.argmin` | (sin args) | - | - | Retorna el indice del elemento con el valor minimo. |
| `tensor.backward` | `gradient` | Tensor | `nil` | Ejecuta la retropropagacion inversa a lo largo del grafo computacional DAG. |

---

### 6. Parametros de Carga de Datos, Persistencia y Utilidades (`GRX::Data`, `GRX::Serialization`, `GRX::Utils`)

* `TensorDataset.new(*tensors)`:
  * `*tensors` (Tensors requeridos): Tensores paralelos (ej. caracteristicas $X$ y etiquetas $Y$) que comparten la dimension 0 de lote.
* `DataLoader.new(dataset, batch_size: 32, shuffle: true)`:
  * `dataset` (`GRX::Data::Dataset`): Coleccion indexable de datos.
  * `batch_size` (Integer, por defecto: `32`): Cantidad de muestras agrupadas que se procesan simultaneamente en C por iteracion antes de actualizar pesos.
  * `shuffle` (Boolean, por defecto: `true`): Si es `true`, desordena aleatoriamente los indices en cada epoca para que la red no aprenda el orden de los datos.
* `GRX::Serialization.save(model, path)` / `model.save_weights(path)`:
  * `model` (`GRX::NN::Module`): Instancia del modelo a persistir.
  * `path` (String): Ruta del archivo destino `.grx`. Vuelca directamente los doubles IEEE 754 de 64 bits de la memoria C sin sobrecosto de JSON/YAML.
* `GRX::Serialization.load(model, path)` / `model.load_weights(path)`:
  * `model` (`GRX::NN::Module`): Modelo con la misma arquitectura en memoria.
  * `path` (String): Ruta del archivo `.grx` binario a cargar.
* `model.train!` y `model.eval!`:
  * `train!`: Pone el modelo en modo de entrenamiento (activa `Dropout` y calcula medias dinamicas en `BatchNorm1d`).
  * `eval!`: Pone el modelo en modo de evaluacion/inferencia (desactiva `Dropout` y usa medias moviles fijas en `BatchNorm1d`).
* `GRX::Utils.clip_grad_norm!(params, max_norm: 1.0)`:
  * `params` (Array[Tensor]): Coleccion de parametros con gradientes acumulados.
  * `max_norm` (Float, por defecto: `1.0`): Norma L2 maxima permitida. Si la norma combinada supera `max_norm`, los gradientes se reescalan proporcionalmente para evitar explosiones.
* `GRX::Utils.one_hot(indices, num_classes: nil, requires_grad: false)`:
  * `indices` (Array[Integer] o Tensor): Vector con identificadores enteros de clase (ej. `[0, 2, 1]`).
  * `num_classes` (Integer, opcional): Numero total de columnas de clase (si no se indica, se autocalcula como `max + 1`).
  * `requires_grad` (Boolean, por defecto: `false`): Si la matriz resultante requiere autograd.
* `GRX.simd_mode`:
  * Retorna el nivel de aceleracion de hardware activo en la maquina: `:avx2` (4 doubles/ciclo con FMA), `:sse` (2 doubles/ciclo), `:scalar` (C portable) o `:ruby` (fallback).

---

### 7. Jerarquia de Excepciones de GRTensor

| Excepcion | Hereda de | Causa Principal |
|---|---|---|
| `GRX::Error` | `StandardError` | Clase base para todas las excepciones del framework. |
| `GRX::ShapeError` | `GRX::Error` | Dimensiones incompatibles en operaciones algebraicas (ej. sumar matrices de distinto tamano o multiplicar dimensiones internas dispares). |
| `GRX::DimensionError` | `GRX::Error` | Rango de dimensiones invalido (ej. llamar `transpose` o `matmul` sobre tensores de 1 sola dimension). |
| `GRX::StorageError` | `GRX::Error` | Fallo de memoria nativa C (`malloc` OOM) o error de lectura en archivos binarios `.grx` corruptos. |

---

## 13. Proyectos Guiados Paso a Paso (Completos y Ejecutables)

---

### Proyecto 1: El Conversor de Temperatura (Celsius a Fahrenheit)

Este es el "Hello World" absoluto de la Inteligencia Artificial. La formula fisica real es:
$$F = 1.8 \times C + 32$$

La red neuronal tiene **1 sola neurona** (`Linear(1, 1)`: $y = w \cdot x + b$). La red no conoce la formula, pero tras 1,500 iteraciones con Adam, descubrira por si sola que el peso $w \approx 1.8$ y el sesgo $b \approx 32.0$.

```ruby
require "grx"

# 1. Datos de entrenamiento (8 pares de temperaturas reales)
celsius_datos    = [18.0, 25.0, 14.0, 21.0,  9.0, 16.0,  4.0, 32.0]
fahrenheit_datos = [64.4, 77.0, 57.2, 69.8, 48.2, 60.8, 39.2, 89.6]

# Convertir a tensores 2D [8 filas, 1 columna]
x = GRX.tensor(celsius_datos, [8, 1])
y = GRX.tensor(fahrenheit_datos, [8, 1])

# 2. Definir la arquitectura (1 neurona)
termometro = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(1, 1)
)

# 3. Optimizador y funcion de error cuadratico medio (MSE)
opt = GRX::Optim::Adam.new(termometro.parameters, lr: 0.8)
loss_fn = GRX::Loss::MSELoss.new

puts "Entrenando la neurona para aprender la escala Fahrenheit..."

# 4. Bucle de entrenamiento
1500.times do
  opt.zero_grad
  pred = termometro.call(x)
  loss = loss_fn.call(pred, y)
  loss.backward
  opt.step
end

puts "Entrenamiento finalizado!"

# 5. Pruebas con temperaturas nunca vistas
test_celsius = GRX.tensor([[100.0], [0.0], [37.0]], [3, 1])
predicciones = termometro.call(test_celsius).to_a

puts "\n--- Resultados del Termometro Neuronal ---"
puts "100.0 C -> #{predicciones[0].round(2)} F (Esperado: 212.00 F)"
puts "  0.0 C -> #{predicciones[1].round(2)} F (Esperado:  32.00 F)"
puts " 37.0 C -> #{predicciones[2].round(2)} F (Esperado:  98.60 F)"
```

---

### Proyecto 2: Compuertas Logicas (AND Lineal vs XOR No Lineal)

Una compuerta **AND** se puede resolver con 1 sola neurona lineal porque los datos son linealmente separables.
Sin embargo, la compuerta **XOR** (Or Exclusivo) es el clasico problema no lineal: una sola linea recta no puede separar los ceros de los unos. Por eso agregamos una **capa oculta con activacion ReLU**:

```ruby
require "grx"

# Tabla de verdad XOR:
# (0, 0) -> 0
# (0, 1) -> 1
# (1, 0) -> 1
# (1, 1) -> 0
entradas_xor = GRX.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [4, 2])
salidas_xor  = GRX.tensor([[0.0], [1.0], [1.0], [0.0]], [4, 1])

# Red neuronal multicapa: 2 entradas -> 4 ocultas (ReLU) -> 1 salida (Sigmoide)
red_xor = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(2, 4),
  GRX::NN::ReLU.new,
  GRX::NN::Linear.new(4, 1),
  GRX::NN::Sigmoid.new
)

opt = GRX::Optim::Adam.new(red_xor.parameters, lr: 0.05)
loss_fn = GRX::Loss::BCELoss.new

# Entrenamiento
400.times do
  opt.zero_grad
  pred = red_xor.call(entradas_xor)
  loss = loss_fn.call(pred, salidas_xor)
  loss.backward
  opt.step
end

puts "\n--- Predicciones de la Compuerta XOR ---"
entradas_xor.to_a.each_slice(2).each_with_index do |(in1, in2), i|
  muestra = GRX.tensor([[in1, in2]], [1, 2])
  resultado = red_xor.call(muestra).item
  puts "XOR(#{in1.to_i}, #{in2.to_i}) = #{resultado.round(3)} -> #{resultado >= 0.5 ? 1 : 0}"
end
```

---

### Proyecto 3: El Predictor de Formulas (Regresion Lineal)

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

### Proyecto 4: Ajuste de Curvas No Lineales y Guardado en `.grx`

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

### Proyecto 5: Clasificador Binario Inteligente con BCELoss

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

### Proyecto 6: Entrenamiento con Datasets Masivos (5,000 Filas con DataLoader)

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

### Proyecto 7: Chatbot Financiero Inteligente con Base de Conocimiento y Filtro de Incertidumbre

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

## 14. Buenas Practicas y Errores Comunes

1. **Llamar `opt.zero_grad` en cada iteracion:** Evita que los gradientes se acumulen indefinidamente.
2. **Dimensiones compatibles:** Las capas lineales esperan siempre tensores 2D `[batch_size, num_features]`.
3. **Manejar `train!` y `eval!`:** Modos necesarios para capas como `Dropout` y `BatchNorm1d` durante inferencia.

---

## 15. Glosario de Terminos Clave

- **Shape:** Dimensiones del tensor (ej. `[10, 4]` -> 10 filas, 4 columnas).
- **Numel:** Total de elementos del tensor.
- **Strides:** Saltos de memoria plana para avanzar en cada eje.
- **Tokenizacion:** Division y conversion de texto a identificadores enteros.
- **Embedding:** Capa que transforma tokens en vectores densos continuos.
- **CrossEntropyLoss:** Perdida multiclase para clasificar texto e intenciones.
- **FMA:** Fused Multiply-Add en CPU que calcula $a \times b + c$ en 1 solo ciclo con minima desviacion.
