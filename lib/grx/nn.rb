# frozen_string_literal: true

module GRX
  module NN
    # ================================================================
    # Module — Base class for all neural network layers
    # ================================================================
    class Module
      # Returns all trainable parameters for optimizer registration
      def parameters
        instance_variables.flat_map do |var|
          val = instance_variable_get(var)
          case val
          when Tensor    then val.requires_grad ? [val] : []
          when Module    then val.parameters
          when Array     then val.flat_map { |v|
            case v
            when Tensor then v.requires_grad ? [v] : []
            when Module then v.parameters
            else []
            end
          }
          else []
          end
        end
      end

      def zero_grad
        parameters.each(&:zero_grad!)
      end

      def save_weights(path)
        GRX::Serialization.save(self, path)
      end

      def load_weights(path)
        GRX::Serialization.load(self, path)
      end

      # Subclasses implement forward computation
      def call(*args)
        forward(*args)
      end
    end

    # ================================================================
    # Linear — Dense fully connected layer
    # y = x @ W^T + b
    # ================================================================
    class Linear < Module
      attr_reader :weight, :bias

      def initialize(in_features, out_features, bias: true)
        @in_features  = in_features
        @out_features = out_features
        @use_bias     = bias

        # Weights: Xavier uniform initialization
        @weight = Tensor.xavier_uniform([out_features, in_features], requires_grad: true)

        # Bias: initialized to zeros
        @bias = bias ? Tensor.zeros([out_features], requires_grad: true) : nil
      end

      def forward(x)
        # x: [batch, in_features]  ->  out: [batch, out_features]
        # out = x @ W^T
        out = x.matmul(@weight.transpose)

        if @use_bias
          batch_size = x.shape[0]
          bias_tiled = _tile_bias(@bias, batch_size, @out_features)
          out + bias_tiled
        else
          out
        end
      end

      private

      def _tile_bias(bias, batch_size, out_features)
        data = Array.new(batch_size) { bias.to_a }.flatten
        tiled = GRX::Tensor.create(data, [batch_size, out_features])
        zero_row = GRX::Tensor.zeros([batch_size, out_features])
        result = zero_row + tiled
        if bias.requires_grad
          result.requires_grad = true
          result._grafo_hijos << bias
          b = bias
          bf = result.backward_fn
          result.backward_fn = ->(g) {
            bf&.call(g)
            grad_data = g.to_a.each_slice(out_features).reduce([0.0] * out_features) { |acc, row|
              acc.zip(row).map { |a, r| a + r }
            }
            b.agregar_gradiente(GRX::Tensor.create(grad_data, [out_features]))
          }
        end
        result
      end

      public

      def to_s
        "Linear(#{@in_features} -> #{@out_features}, bias: #{@use_bias})"
      end
    end

    # ================================================================
    # Sequential — Chains layers sequentially in order
    # ================================================================
    class Sequential < Module
      def initialize(*layers)
        @layers = layers
      end

      def forward(x)
        @layers.reduce(x) { |input, layer| layer.call(input) }
      end

      def parameters
        @layers.flat_map(&:parameters)
      end

      def train!
        @layers.each { |l| l.train! if l.respond_to?(:train!) }
        self
      end

      def eval!
        @layers.each { |l| l.eval! if l.respond_to?(:eval!) }
        self
      end

      def to_s
        layers_str = @layers.each_with_index.map { |l, i| "  (#{i}): #{l}" }.join("\n")
        "Sequential(\n#{layers_str}\n)"
      end
    end

    # ================================================================
    # Activation layers (for use inside Sequential pipelines)
    # ================================================================
    class ReLU < Module
      def forward(x) = x.relu
      def to_s = "ReLU()"
    end

    class LeakyReLU < Module
      attr_reader :alpha

      def initialize(alpha_arg = nil, alpha: nil)
        @alpha = (alpha || alpha_arg || 0.01).to_f
      end
      def forward(x) = x.leaky_relu(@alpha)
      def to_s = "LeakyReLU(alpha=#{@alpha})"
    end

    class Tanh < Module
      def forward(x) = x.tanh
      def to_s = "Tanh()"
    end

    class Sigmoid < Module
      def forward(x) = x.sigmoid
      def to_s = "Sigmoid()"
    end

    class Softmax < Module
      def forward(x) = x.softmax
      def to_s = "Softmax()"
    end

    # ================================================================
    # Dropout — Random feature dropout during training mode
    # ================================================================
    class Dropout < Module
      def initialize(p = 0.5)
        @p        = p
        @training = true
      end

      def train!;  @training = true;  self; end
      def eval!;   @training = false; self; end

      def forward(x)
        return x unless @training && @p > 0

        scale = 1.0 / (1.0 - @p)
        mask_data = x.to_a.map { rand > @p ? scale : 0.0 }
        mask = Tensor.create(mask_data, x.shape)
        x * mask
      end

      def to_s = "Dropout(p=#{@p})"
    end

    # ================================================================
    # Embedding — Dense vector lookup table for token indices
    # ================================================================
    class Embedding < Module
      attr_reader :weight, :num_embeddings, :embedding_dim

      def initialize(num_embeddings, embedding_dim)
        @num_embeddings = num_embeddings
        @embedding_dim  = embedding_dim
        @weight         = Tensor.he_normal([num_embeddings, embedding_dim], requires_grad: true)
      end

      def forward(indices)
        ids = indices.is_a?(Tensor) ? indices.to_a.map(&:to_i) : indices.map(&:to_i)
        batch_size = ids.size
        out_data = ids.flat_map do |id|
          raise IndexError, "Token index #{id} out of range [0, #{@num_embeddings})" if id < 0 || id >= @num_embeddings
          @weight.to_a.slice(id * @embedding_dim, @embedding_dim)
        end

        out = Tensor.create(out_data, [batch_size, @embedding_dim])
        if @weight.requires_grad
          out.requires_grad = true
          out._grafo_hijos << @weight
          w = @weight; dim = @embedding_dim; num_emb = @num_embeddings
          out.backward_fn = ->(g) {
            grad_w = Array.new(num_emb * dim, 0.0)
            g_data = g.to_a
            ids.each_with_index do |id, i|
              slice = g_data.slice(i * dim, dim)
              dim.times { |d| grad_w[id * dim + d] += slice[d] }
            end
            w.agregar_gradiente(Tensor.create(grad_w, w.shape))
          }
        end
        out
      end

      def to_s
        "Embedding(#{@num_embeddings}, #{@embedding_dim})"
      end
    end

    # ================================================================
    # LayerNorm — Layer normalization across channel dimensions
    # ================================================================
    class LayerNorm < Module
      attr_reader :gamma, :beta, :normalized_shape, :epsilon

      def initialize(normalized_shape, eps: nil, epsilon: 1e-5)
        @normalized_shape = normalized_shape.is_a?(Array) ? normalized_shape : [normalized_shape]
        @dim              = @normalized_shape.reduce(1, :*)
        @epsilon          = (eps || epsilon).to_f

        @gamma = Tensor.ones(@normalized_shape, requires_grad: true)
        @beta  = Tensor.zeros(@normalized_shape, requires_grad: true)
      end

      def forward(x)
        batch_size = x.shape[0]
        x_data = x.to_a

        means = Array.new(batch_size) do |b|
          x_data.slice(b * @dim, @dim).sum / @dim.to_f
        end
        vars = Array.new(batch_size) do |b|
          m = means[b]
          x_data.slice(b * @dim, @dim).sum { |v| (v - m)**2 } / @dim.to_f
        end

        gamma_data = @gamma.to_a
        beta_data  = @beta.to_a
        norm_data  = Array.new(batch_size * @dim)

        batch_size.times do |b|
          m = means[b]
          inv_std = 1.0 / Math.sqrt(vars[b] + @epsilon)
          @dim.times do |j|
            norm_data[b * @dim + j] = gamma_data[j] * (x_data[b * @dim + j] - m) * inv_std + beta_data[j]
          end
        end

        out = Tensor.create(norm_data, x.shape)
        if x.requires_grad || @gamma.requires_grad || @beta.requires_grad
          out.requires_grad = true
          out._grafo_hijos.push(x, @gamma, @beta)
          g_param = @gamma; b_param = @beta; d = @dim; eps = @epsilon
          out.backward_fn = ->(g) {
            g_data = g.to_a
            grad_gamma = Array.new(d, 0.0)
            grad_beta  = Array.new(d, 0.0)
            grad_x     = Array.new(batch_size * d, 0.0)

            batch_size.times do |b|
              m = means[b]; v = vars[b]
              inv_std = 1.0 / Math.sqrt(v + eps)
              x_hat = Array.new(d) { |j| (x_data[b * d + j] - m) * inv_std }
              dl_dxhat = Array.new(d) { |j| g_data[b * d + j] * gamma_data[j] }
              sum_dl = dl_dxhat.sum
              sum_dl_x = dl_dxhat.zip(x_hat).sum { |a, c| a * c }

              d.times do |j|
                grad_gamma[j] += g_data[b * d + j] * x_hat[j]
                grad_beta[j]  += g_data[b * d + j]
                grad_x[b * d + j] = (inv_std / d.to_f) * (d.to_f * dl_dxhat[j] - sum_dl - x_hat[j] * sum_dl_x)
              end
            end

            g_param.agregar_gradiente(Tensor.create(grad_gamma, g_param.shape)) if g_param.requires_grad
            b_param.agregar_gradiente(Tensor.create(grad_beta, b_param.shape))   if b_param.requires_grad
            x.agregar_gradiente(Tensor.create(grad_x, x.shape))                 if x.requires_grad
          }
        end
        out
      end

      def to_s
        "LayerNorm(#{@normalized_shape})"
      end
    end

    # ================================================================
    # BatchNorm1d — Normalizacion por batch
    # ================================================================
    class BatchNorm1d < Module
      def initialize(num_features, eps: nil, epsilon: 1e-5, momentum: 0.1)
        @num_features = num_features
        @epsilon      = (eps || epsilon).to_f
        @momentum     = momentum.to_f
        @training     = true

        @gamma = Tensor.ones([num_features],  requires_grad: true)
        @beta  = Tensor.zeros([num_features], requires_grad: true)

        @running_mean = Tensor.zeros([num_features])
        @running_var  = Tensor.ones([num_features])
      end

      def train!; @training = true;  self; end
      def eval!;  @training = false; self; end

      def forward(x)
        batch_size = x.shape[0]

        if @training
          batch_data = x.to_a
          means = Array.new(@num_features) do |j|
            batch_data.each_slice(@num_features).map { |row| row[j] }.sum / batch_size
          end
          vars = Array.new(@num_features) do |j|
            col = batch_data.each_slice(@num_features).map { |row| row[j] }
            col.sum { |v| (v - means[j]) ** 2 } / batch_size
          end

          means.each_with_index do |m, j|
            rm = @running_mean.to_a; rm[j] = (1 - @momentum) * rm[j] + @momentum * m
            @running_mean = Tensor.create(rm, [@num_features])
          end
          vars.each_with_index do |v, j|
            rv = @running_var.to_a; rv[j] = (1 - @momentum) * rv[j] + @momentum * v
            @running_var = Tensor.create(rv, [@num_features])
          end

          mean_t = Tensor.create(means, [@num_features])
          var_t  = Tensor.create(vars,  [@num_features])
        else
          mean_t = @running_mean
          var_t  = @running_var
        end

        norm_data = x.to_a.each_slice(@num_features).flat_map do |row|
          row.each_with_index.map do |v, j|
            x_hat = (v - mean_t.to_a[j]) / Math.sqrt(var_t.to_a[j] + @epsilon)
            @gamma.to_a[j] * x_hat + @beta.to_a[j]
          end
        end

        Tensor.create(norm_data, x.shape)
      end

      def to_s = "BatchNorm1d(#{@num_features})"
    end
  end
end
