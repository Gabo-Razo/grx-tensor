# frozen_string_literal: true

module GRX
  module Optim
    # ================================================================
    # SGD — Stochastic Gradient Descent (with optional momentum)
    # ================================================================
    class SGD
      def initialize(params, lr: 0.01, momentum: 0.0, weight_decay: 0.0)
        @params       = params
        @lr           = lr
        @momentum     = momentum
        @weight_decay = weight_decay
        # Velocity buffer for momentum
        @velocity = params.map { |p| Tensor.zeros_like(p) }
      end

      def step
        @params.each_with_index do |param, i|
          next unless param.grad

          grad = param.grad

          # L2 regularization (weight decay)
          if @weight_decay > 0
            grad = grad + param.scale(@weight_decay)
          end

          if @momentum > 0
            # v = momentum*v + grad
            @velocity[i] = @velocity[i].scale(@momentum) + grad
            grad = @velocity[i]
          end

          if CAPI::LOADED
            CAPI.grx_sgd_step(param.storage.ptr, grad.storage.ptr, @lr, param.numel)
          else
            # Ruby fallback
            param_data = param.to_a
            grad_data  = grad.to_a
            param_data.each_with_index { |v, j| param_data[j] = v - @lr * grad_data[j] }
            param.storage.instance_variable_set(:@data, param_data)
          end
        end
      end

      def zero_grad
        @params.each(&:zero_grad!)
      end
    end

    # ================================================================
    # Adam — Adaptive Moment Estimation (Kingma & Ba, 2015)
    # The standard optimizer for deep neural networks.
    # ================================================================
    class Adam
      def initialize(params, lr: 0.001, beta1: 0.9, beta2: 0.999,
                     epsilon: 1e-8, weight_decay: 0.0)
        @params       = params
        @lr           = lr
        @beta1        = beta1
        @beta2        = beta2
        @epsilon      = epsilon
        @weight_decay = weight_decay
        @t            = 0  # current step

        # First and second order moment vectors (zero-initialized)
        @m = params.map { |p| Tensor.zeros_like(p) }
        @v = params.map { |p| Tensor.zeros_like(p) }
      end

      def step
        @t += 1
        beta1t = @beta1 ** @t  # beta1^t for bias correction
        beta2t = @beta2 ** @t

        @params.each_with_index do |param, i|
          next unless param.grad

          grad = param.grad

          if @weight_decay > 0
            grad = grad + param.scale(@weight_decay)
          end

          if CAPI::LOADED
            CAPI.grx_adam_step(
              param.storage.ptr,
              @m[i].storage.ptr,
              @v[i].storage.ptr,
              grad.storage.ptr,
              @lr, @beta1, @beta2, @epsilon,
              beta1t, beta2t,
              param.numel
            )
          else
            # Pure Ruby fallback
            p_data = param.to_a
            m_data = @m[i].to_a
            v_data = @v[i].to_a
            g_data = grad.to_a
            p_data.each_with_index do |_, j|
              m_data[j] = @beta1 * m_data[j] + (1 - @beta1) * g_data[j]
              v_data[j] = @beta2 * v_data[j] + (1 - @beta2) * g_data[j] ** 2
              mh = m_data[j] / (1 - beta1t)
              vh = v_data[j] / (1 - beta2t)
              p_data[j] -= @lr * mh / (Math.sqrt(vh) + @epsilon)
            end
            param.storage.instance_variable_set(:@data, p_data)
            @m[i].storage.instance_variable_set(:@data, m_data)
            @v[i].storage.instance_variable_set(:@data, v_data)
          end
        end
      end

      def zero_grad
        @params.each(&:zero_grad!)
      end
    end
  end
end
