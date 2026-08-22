# frozen_string_literal: true

module GRX
  class Tensor
    attr_reader   :storage, :shape, :strides, :offset
    attr_accessor :grad, :requires_grad, :backward_fn

    def initialize(storage, shape, strides: nil, offset: 0, requires_grad: false)
      @storage       = storage
      @shape         = shape
      @offset        = offset
      @strides       = strides || _calc_strides(shape)
      @requires_grad = requires_grad
      @grad          = nil
      @backward_fn   = nil
      @_grafo_hijos  = []
    end

    # ----------------------------------------------------------------
    # FACTORIES
    # ----------------------------------------------------------------

    def self.create(array_valores, shape, requires_grad: false)
      new(Storage.new(array_valores), shape, requires_grad: requires_grad)
    end

    def self.zeros(shape, requires_grad: false)
      create(Array.new(shape.reduce(1,:*), 0.0), shape, requires_grad: requires_grad)
    end

    def self.ones(shape, requires_grad: false)
      create(Array.new(shape.reduce(1,:*), 1.0), shape, requires_grad: requires_grad)
    end

    def self.zeros_like(t, requires_grad: false)
      zeros(t.shape, requires_grad: requires_grad)
    end

    def self.ones_like(t, requires_grad: false)
      ones(t.shape, requires_grad: requires_grad)
    end

    # Xavier uniform initialization (optimal for linear layers with tanh/sigmoid)
    def self.xavier_uniform(shape, requires_grad: false)
      fan_in, fan_out = shape[-2] || 1, shape[-1] || 1
      n = shape.reduce(1, :*)
      s = _alloc_raw(n)
      CAPI.grx_init_xavier_uniform(s.ptr, n, fan_in, fan_out) if CAPI::LOADED
      new(s, shape, requires_grad: requires_grad)
    end

    # He normal initialization (optimal for layers with ReLU)
    def self.he_normal(shape, requires_grad: false)
      # fan_in = number of inputs = last dim or penultimate if 2D
      fan_in = shape.size >= 2 ? shape[-1] : shape[0]
      n = shape.reduce(1, :*)
      s = _alloc_raw(n)
      CAPI.grx_init_he_normal(s.ptr, n, fan_in) if CAPI::LOADED
      new(s, shape, requires_grad: requires_grad)
    end

    # ----------------------------------------------------------------
    # ARITHMETIC OPERATIONS (with autograd)
    # ----------------------------------------------------------------

    def +(other)
      case other
      when Tensor
        raise ShapeError, "Incompatible shapes: #{@shape} vs #{other.shape}" if @shape != other.shape
        r = Tensor.new(_binop(:grx_add, other), @shape)
        if requires_grad || other.requires_grad
          r.requires_grad = true
          r._grafo_hijos.push(self, other)
          r.backward_fn = ->(g) {
            agregar_gradiente(g)       if requires_grad
            other.agregar_gradiente(g) if other.requires_grad
          }
        end
        r
      when Numeric
        add_scalar(other.to_f)
      else
        raise TypeError, "Cannot add Tensor with #{other.class}"
      end
    end

    def -(other)
      case other
      when Tensor
        raise ShapeError, "Incompatible shapes: #{@shape} vs #{other.shape}" if @shape != other.shape
        r = Tensor.new(_binop(:grx_sub, other), @shape)
        if requires_grad || other.requires_grad
          r.requires_grad = true
          r._grafo_hijos.push(self, other)
          r.backward_fn = ->(g) {
            agregar_gradiente(g)              if requires_grad
            other.agregar_gradiente(g.negate) if other.requires_grad
          }
        end
        r
      when Numeric
        add_scalar(-other.to_f)
      else
        raise TypeError, "Cannot subtract Tensor with #{other.class}"
      end
    end

    def *(other)
      case other
      when Tensor
        raise ShapeError, "Incompatible shapes: #{@shape} vs #{other.shape}" if @shape != other.shape
        r = Tensor.new(_binop(:grx_mul, other), @shape)
        if requires_grad || other.requires_grad
          r.requires_grad = true
          a, b = self, other
          r._grafo_hijos.push(a, b)
          r.backward_fn = ->(g) {
            a.agregar_gradiente(g * b) if a.requires_grad
            b.agregar_gradiente(g * a) if b.requires_grad
          }
        end
        r
      when Numeric
        scale(other.to_f)
      else
        raise TypeError, "Cannot multiply Tensor with #{other.class}"
      end
    end

    def /(other)
      case other
      when Tensor
        raise ShapeError, "Incompatible shapes: #{@shape} vs #{other.shape}" if @shape != other.shape
        r = Tensor.new(_binop(:grx_div, other), @shape)
        if requires_grad || other.requires_grad
          r.requires_grad = true
          a, b = self, other
          r._grafo_hijos.push(a, b)
          r.backward_fn = ->(g) {
            # d(a/b)/da = 1/b,  d(a/b)/db = -a/b^2
            a.agregar_gradiente(g / b)                    if a.requires_grad
            b.agregar_gradiente((g * a).negate / (b * b)) if b.requires_grad
          }
        end
        r
      when Numeric
        scale(1.0 / other.to_f)
      else
        raise TypeError, "Cannot divide Tensor with #{other.class}"
      end
    end

    def -@
      negate
    end

    # ----------------------------------------------------------------
    # SCALAR OPERATIONS
    # ----------------------------------------------------------------

    def coerce(other)
      case other
      when Numeric
        # Returns reversed [self, other] wrapper to enable 2.0 * tensor
        [Tensor.new(Storage.new(Array.new(numel, other.to_f)), @shape), self]
      else
        raise TypeError, "#{self.class} cannot be coerced with #{other.class}"
      end
    end

    def scale(s)
      r = _unary_c(:grx_scale, s) { |v| v * s }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self; factor = s.to_f
        r.backward_fn = ->(g) { src.agregar_gradiente(g.scale(factor)) }
      end
      r
    end

    def add_scalar(s)
      r = _unary_c(:grx_add_scalar, s) { |v| v + s }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self
        r.backward_fn = ->(g) { src.agregar_gradiente(g) }
      end
      r
    end

    def negate
      r = _unary_c(:grx_negate) { |v| -v }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self
        r.backward_fn = ->(g) { src.agregar_gradiente(g.negate) }
      end
      r
    end

    # ----------------------------------------------------------------
    # ELEMENT-WISE MATH (with autograd)
    # ----------------------------------------------------------------

    def abs
      r = _unary_c(:grx_abs) { |v| v.abs }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self
        r.backward_fn = ->(g) {
          # d|x|/dx = sign(x)
          sign = Tensor.create(src.to_a.map { |v| v >= 0 ? 1.0 : -1.0 }, src.shape)
          src.agregar_gradiente(g * sign)
        }
      end
      r
    end

    def sqrt
      r = _unary_c(:grx_sqrt) { |v| Math.sqrt(v) }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        res = r; src = self
        r.backward_fn = ->(g) {
          # d(sqrt(x))/dx = 1 / (2*sqrt(x))
          src.agregar_gradiente(g / (res.scale(2.0)))
        }
      end
      r
    end

    def square
      r = _unary_c(:grx_square) { |v| v * v }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self
        r.backward_fn = ->(g) { src.agregar_gradiente(g * src.scale(2.0)) }
      end
      r
    end

    def log
      r = _unary_c(:grx_log) { |v| Math.log(v) }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self
        r.backward_fn = ->(g) { src.agregar_gradiente(g / src) }
      end
      r
    end

    def exp
      r = _unary_c(:grx_exp) { |v| Math.exp(v) }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        res = r; src = self
        r.backward_fn = ->(g) { src.agregar_gradiente(g * res) }
      end
      r
    end

    def pow(e)
      r = _unary_c(:grx_pow, e.to_f) { |v| v ** e }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self
        r.backward_fn = ->(g) {
          src.agregar_gradiente(g * src.pow(e - 1).scale(e.to_f))
        }
      end
      r
    end

    def clip(lo, hi)
      out = _alloc_storage(numel)
      if CAPI::LOADED
        CAPI.grx_clip(@storage.ptr, lo.to_f, hi.to_f, out.ptr, numel)
      else
        data = to_a.map { |v| v < lo ? lo : (v > hi ? hi : v) }
        return Tensor.create(data, @shape, requires_grad: @requires_grad)
      end
      r = Tensor.new(out, @shape)
      if @requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self; l = lo.to_f; h = hi.to_f
        r.backward_fn = ->(g) {
          mask = Tensor.create(src.to_a.map { |v| (v >= l && v <= h) ? 1.0 : 0.0 }, src.shape)
          src.agregar_gradiente(g * mask)
        }
      end
      r
    end

    # ----------------------------------------------------------------
    # REDUCTIONS (return differentiable scalar Tensor with autograd)
    # ----------------------------------------------------------------

    def sum
      val = if CAPI::LOADED
        CAPI.grx_sum(@storage.ptr, numel)
      else
        to_a.sum
      end
      r = Tensor.create([val], [1], requires_grad: @requires_grad)
      if @requires_grad
        r._grafo_hijos << self
        src = self
        r.backward_fn = ->(g) {
          src.agregar_gradiente(Tensor.create(Array.new(src.numel, g.item), src.shape))
        }
      end
      r
    end

    def mean
      val = if CAPI::LOADED
        CAPI.grx_mean(@storage.ptr, numel)
      else
        to_a.sum.to_f / numel
      end
      r = Tensor.create([val], [1], requires_grad: @requires_grad)
      if @requires_grad
        r._grafo_hijos << self
        src = self; n = numel.to_f
        r.backward_fn = ->(g) {
          src.agregar_gradiente(Tensor.create(Array.new(src.numel, g.item / n), src.shape))
        }
      end
      r
    end

    def max
      if CAPI::LOADED
        CAPI.grx_max(@storage.ptr, numel)
      else
        to_a.max
      end
    end

    def min
      if CAPI::LOADED
        CAPI.grx_min(@storage.ptr, numel)
      else
        to_a.min
      end
    end

    # ----------------------------------------------------------------
    # LINEAR ALGEBRA
    # ----------------------------------------------------------------

    def dot(other)
      raise ShapeError, "dot requires matching shape" if @shape != other.shape
      if CAPI::LOADED
        CAPI.grx_dot(@storage.ptr, other.storage.ptr, numel)
      else
        to_a.zip(other.to_a).sum { |a, b| a * b }
      end
    end

    def matmul(other)
      raise DimensionError, "matmul requires 2D tensors" unless @shape.size == 2 && other.shape.size == 2
      m, k = @shape; k2, n = other.shape
      raise ShapeError, "Incompatible dimensions: #{@shape} × #{other.shape}" if k != k2
      out = _alloc_storage(m * n)
      if CAPI::LOADED
        CAPI.grx_matmul(@storage.ptr, other.storage.ptr, out.ptr, m, k, n)
      else
        result = Array.new(m * n, 0.0)
        m.times { |i| k.times { |kk| aik = @storage.read(i*k+kk)
          n.times { |j| result[i*n+j] += aik * other.storage.read(kk*n+j) } } }
        return Tensor.create(result, [m, n])
      end
      r = Tensor.new(out, [m, n])
      if requires_grad || other.requires_grad
        r.requires_grad = true
        a, b = self, other
        r._grafo_hijos.push(a, b)
        r.backward_fn = ->(g) {
          # dL/dA = dL/dC × B^T,  dL/dB = A^T × dL/dC
          # Uses _matmul_no_grad and _transpose_view to avoid graph recursion
          a.agregar_gradiente(g._matmul_no_grad(b._transpose_view)) if a.requires_grad
          b.agregar_gradiente(a._transpose_view._matmul_no_grad(g)) if b.requires_grad
        }
      end
      r
    end

    # ----------------------------------------------------------------
    # ACTIVATIONS (with autograd)
    # ----------------------------------------------------------------

    def relu
      r = _unary_c(:grx_relu) { |v| v > 0 ? v : 0.0 }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self
        r.backward_fn = ->(g) {
          mask = Tensor.create(src.to_a.map { |v| v > 0 ? 1.0 : 0.0 }, src.shape)
          src.agregar_gradiente(g * mask)
        }
      end
      r
    end

    def leaky_relu(alpha = 0.01)
      r = _unary_c(:grx_leaky_relu, alpha.to_f) { |v| v > 0 ? v : alpha * v }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        src = self
        r.backward_fn = ->(g) {
          mask = Tensor.create(src.to_a.map { |v| v > 0 ? 1.0 : alpha }, src.shape)
          src.agregar_gradiente(g * mask)
        }
      end
      r
    end

    def tanh
      r = _unary_c(:grx_tanh_act) { |v| Math.tanh(v) }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        res = r; src = self
        r.backward_fn = ->(g) {
          # d(tanh)/dx = 1 - tanh(x)^2
          src.agregar_gradiente(g * (Tensor.ones_like(res) - res.square))
        }
      end
      r
    end

    def sigmoid
      r = _unary_c(:grx_sigmoid) { |v| 1.0 / (1.0 + Math.exp(-v)) }
      if requires_grad
        r.requires_grad = true; r._grafo_hijos << self
        res = r; src = self
        r.backward_fn = ->(g) {
          # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
          src.agregar_gradiente(g * res * (Tensor.ones_like(res) - res))
        }
      end
      r
    end

    def softmax
      dim = @shape[-1]
      batch = numel / dim
      raw = to_a
      out_vals = Array.new(numel)

      batch.times do |b|
        slice = raw.slice(b * dim, dim)
        max_v = slice.max
        exps = slice.map { |v| Math.exp(v - max_v) }
        sum_e = exps.sum
        dim.times { |j| out_vals[b * dim + j] = exps[j] / sum_e }
      end

      r = Tensor.create(out_vals, @shape, requires_grad: @requires_grad)
      if @requires_grad
        r._grafo_hijos << self
        res = r; src = self
        r.backward_fn = ->(g) {
          s_data = res.to_a
          g_data = g.to_a
          grad_x = Array.new(src.numel, 0.0)

          batch.times do |b|
            s_row = s_data.slice(b * dim, dim)
            g_row = g_data.slice(b * dim, dim)
            dot = s_row.zip(g_row).sum { |s_val, g_val| s_val * g_val }
            dim.times do |j|
              grad_x[b * dim + j] = s_row[j] * (g_row[j] - dot)
            end
          end

          src.agregar_gradiente(Tensor.create(grad_x, src.shape))
        }
      end
      r
    end

    # ----------------------------------------------------------------
    # AUTOGRAD
    # ----------------------------------------------------------------

    def agregar_gradiente(g)
      @grad = @grad.nil? ? g : @grad + g
    end

    def backward(grad_inicial = nil)
      if grad_inicial.nil? && @grad.nil?
        agregar_gradiente(Tensor.ones(@shape))
      elsif !grad_inicial.nil?
        agregar_gradiente(grad_inicial)
      end

      # Topological sorting via iterative post-order DFS (prevents stack overflow on deep graphs)
      orden     = []
      visitados = {}
      stack     = [[self, false]]

      until stack.empty?
        nodo, procesado = stack.pop
        if procesado
          orden << nodo unless visitados[nodo.object_id]
          visitados[nodo.object_id] = true
        else
          next if visitados[nodo.object_id]
          stack.push([nodo, true])
          nodo._grafo_hijos.each { |h| stack.push([h, false]) unless visitados[h.object_id] }
        end
      end

      # Topological order in post-order: reverse traverses root first down to leaves
      orden.reverse_each do |nodo|
        next unless nodo.grad && nodo.backward_fn
        nodo.backward_fn.call(nodo.grad)
        nodo.backward_fn = nil
      end
    end

    def zero_grad!
      @grad = nil
      @_grafo_hijos = []
      @backward_fn = nil
    end

    def _grafo_hijos
      @_grafo_hijos
    end

    # ----------------------------------------------------------------
    # GEOMETRY (zero-copy)
    # ----------------------------------------------------------------

    def get(*coords)
      @storage.read(_calc_flat_index(coords))
    end

    def contiguous
      return self if _contiguous?
      c = Tensor.create(to_a, @shape, requires_grad: @requires_grad)
      if @requires_grad
        c._grafo_hijos << self
        src = self
        c.backward_fn = ->(g) { src.agregar_gradiente(g) }
      end
      c
    end

    def reshape(nueva_forma)
      raise ArgumentError, "Incompatible reshape" if numel != nueva_forma.reduce(1,:*)
      r = Tensor.new(@storage, nueva_forma, offset: @offset, requires_grad: @requires_grad)
      if @requires_grad
        r._grafo_hijos << self
        src = self; orig_shape = @shape
        r.backward_fn = ->(g) { src.agregar_gradiente(g.reshape(orig_shape)) }
      end
      r
    end

    def transpose
      raise DimensionError, "transpose only supports 2D tensors" if @shape.size != 2
      t = Tensor.new(@storage, [@shape[1], @shape[0]],
                 strides: [@strides[1], @strides[0]],
                 offset: @offset, requires_grad: @requires_grad)
      if @requires_grad
        t._grafo_hijos << self
        src = self
        t.backward_fn = ->(g) {
          src.agregar_gradiente(g.transpose)
        }
      end
      t
    end

    # Transpose view without autograd — for internal backward pass
    def _transpose_view
      raise DimensionError, "transpose only supports 2D tensors" if @shape.size != 2
      Tensor.new(@storage, [@shape[1], @shape[0]],
                 strides: [@strides[1], @strides[0]],
                 offset: @offset, requires_grad: false)
    end

    # Matmul without autograd — for internal backward_fn usage
    def _matmul_no_grad(other)
      raise DimensionError, "matmul requires 2D tensors" unless @shape.size == 2 && other.shape.size == 2
      m, k = @shape; k2, n = other.shape
      raise ShapeError, "Incompatible dimensions" if k != k2
      a_c = _contiguous? ? self : contiguous
      b_c = other._contiguous? ? other : other.contiguous
      out = _alloc_storage(m * n)
      if CAPI::LOADED
        CAPI.grx_matmul(a_c.storage.ptr, b_c.storage.ptr, out.ptr, m, k, n)
      else
        result = Array.new(m * n, 0.0)
        m.times { |i| k.times { |kk| aik = a_c.storage.read(i*k+kk)
          n.times { |j| result[i*n+j] += aik * b_c.storage.read(kk*n+j) } } }
        return Tensor.new(Storage.new(result), [m, n])
      end
      Tensor.new(out, [m, n])
    end

    def flatten
      reshape([numel])
    end

    # ----------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------

    def numel
      @shape.reduce(1, :*)
    end

    def to_a
      # If strides are contiguous (normal tensor, reshape), read buffer directly.
      # Otherwise (transpose, strided views), traverse with custom strides.
      if _contiguous?
        @storage.to_ruby_array
      else
        _collect_elements(@shape, @strides, @offset)
      end
    end

    # A tensor is contiguous if its strides match standard row-major order
    def contiguous?
      expected = _calc_strides(@shape)
      @strides == expected && @offset == 0
    end
    alias _contiguous? contiguous?

    private

    def _collect_elements(shape, strides, offset)
      if shape.size == 1
        Array.new(shape[0]) { |i| @storage.read(offset + i * strides[0]) }
      else
        Array.new(shape[0]) { |i|
          _collect_elements(shape[1..], strides[1..], offset + i * strides[0])
        }.flatten
      end
    end

    public

    include Comparable

    def <=>(other)
      case other
      when Tensor
        (numel == 1 && other.numel == 1) ? item <=> other.item : nil
      when Numeric
        numel == 1 ? item <=> other.to_f : nil
      else
        nil
      end
    end

    def item
      raise "item() only supported for 1-element tensors" if numel != 1
      to_a[0]
    end

    def to_f
      raise "to_f only supported for 1-element tensors" if numel != 1
      to_a[0]
    end

    def to_i
      raise "to_i only supported for 1-element tensors" if numel != 1
      to_a[0].to_i
    end

    def nan?
      raise "nan? only supported for 1-element tensors" if numel != 1
      to_a[0].nan?
    end

    def to_s
      "#<GRX::Tensor shape=#{@shape} data=#{to_a}>"
    end
    alias inspect to_s

    # ----------------------------------------------------------------
    # PRIVATE
    # ----------------------------------------------------------------

    private

    def _alloc_storage(n)
      self.class._alloc_raw(n)
    end

    def self._alloc_raw(n)
      if CAPI::LOADED
        ptr = CAPI.grx_alloc(n)
        raise StorageError, "grx_alloc OOM" if ptr.null?
        s = Storage.allocate
        s.instance_variable_set(:@size, n)
        s.instance_variable_set(:@ptr,  ptr)
        ObjectSpace.define_finalizer(s, Storage.make_finalizer(ptr))
        s
      else
        Storage.new(Array.new(n, 0.0))
      end
    end

    # Element-wise binary op: delegates to CAPI or Ruby fallback
    def _binop(op, other)
      a_c = _contiguous? ? self : contiguous
      b_c = other._contiguous? ? other : other.contiguous
      out = _alloc_storage(numel)
      if CAPI::LOADED
        CAPI.public_send(op, a_c.storage.ptr, b_c.storage.ptr, out.ptr, numel)
      else
        rb = { grx_add: :+, grx_sub: :-, grx_mul: :*, grx_div: :/ }[op]
        data = (0...numel).map { |i|
          a_c.storage.read(a_c.offset + i).public_send(rb, b_c.storage.read(b_c.offset + i))
        }
        return Storage.new(data)
      end
      out
    end

    # Unary op: delegates to CAPI with optional args or Ruby fallback block
    def _unary_c(op, *args, &fallback)
      a_c = _contiguous? ? self : contiguous
      out = _alloc_storage(numel)
      if CAPI::LOADED
        CAPI.public_send(op, a_c.storage.ptr, *args, out.ptr, numel)
      else
        vals = if fallback
          fallback.arity == 0 ? fallback.call : a_c.to_a.map(&fallback)
        else
          a_c.to_a
        end
        return Tensor.create(vals, @shape)
      end
      Tensor.new(out, @shape)
    end

    def _calc_flat_index(coords)
      idx = @offset
      coords.each_with_index { |c, i| idx += c * @strides[i] }
      idx
    end

    def _calc_strides(shape)
      s = 1; saltos = []
      shape.reverse_each { |d| saltos.unshift(s); s *= d }
      saltos
    end
  end
end
