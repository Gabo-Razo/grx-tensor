# frozen_string_literal: true

module GRX
  module Loss
    # ================================================================
    # MSELoss — Mean Squared Error
    # L = mean((pred - target)^2)   -> returns differentiable scalar Tensor
    # ================================================================
    class MSELoss
      def call(pred, target)
        raise ShapeError, "Incompatible shapes: #{pred.shape} vs #{target.shape}" if pred.shape != target.shape
        (pred - target).square.mean
      end
    end

    # ================================================================
    # MAELoss — Mean Absolute Error
    # L = mean(|pred - target|)
    # ================================================================
    class MAELoss
      def call(pred, target)
        raise ShapeError, "Incompatible shapes: #{pred.shape} vs #{target.shape}" if pred.shape != target.shape
        (pred - target).abs.mean
      end
    end

    # ================================================================
    # BCELoss — Binary Cross-Entropy
    # L = -mean(t*log(p) + (1-t)*log(1-p))
    # pred must be in (0,1) — apply Sigmoid before if using logits.
    # ================================================================
    class BCELoss
      EPS = 1e-7

      def call(pred, target)
        raise ShapeError, "Incompatible shapes: #{pred.shape} vs #{target.shape}" if pred.shape != target.shape
        p_clamped = pred.clip(EPS, 1.0 - EPS)
        ones = Tensor.ones_like(target)
        term1 = target * p_clamped.log
        term2 = (ones - target) * (ones - p_clamped).log
        (-(term1 + term2)).mean
      end
    end

    # ================================================================
    # CrossEntropyLoss — Softmax + NLL (multi-class)
    # L = -sum(target * log(softmax(logits))) / batch_size
    # ================================================================
    class CrossEntropyLoss
      EPS = 1e-7

      def call(logits, target)
        raise ShapeError, "Incompatible shapes: #{logits.shape} vs #{target.shape}" if logits.shape != target.shape
        probs = logits.softmax
        p_data = probs.to_a
        t_data = target.to_a
        batch_size = logits.shape[0].to_f

        loss_val = t_data.each_with_index.sum do |t, i|
          next 0.0 if t == 0.0
          p = [p_data[i], EPS].max
          -t * Math.log(p)
        end / batch_size

        out = Tensor.create([loss_val], [1], requires_grad: logits.requires_grad || target.requires_grad)
        if logits.requires_grad || target.requires_grad
          out._grafo_hijos.push(logits, target)
          out.backward_fn = ->(g) {
            scale = g.item / batch_size
            grad_logits = p_data.zip(t_data).map { |p, t| (p - t) * scale }
            logits.agregar_gradiente(Tensor.create(grad_logits, logits.shape)) if logits.requires_grad
          }
        end
        out
      end
    end

    # ================================================================
    # HuberLoss — Smooth L1 (robust against outliers)
    # ================================================================
    class HuberLoss
      attr_reader :delta

      def initialize(delta: 1.0)
        @delta = delta.to_f
      end

      def call(pred, target)
        raise ShapeError, "Shapes incompatibles: #{pred.shape} vs #{target.shape}" if pred.shape != target.shape
        diff_data = (pred - target).abs.to_a
        d = @delta
        loss_val = diff_data.sum { |v| v <= d ? 0.5 * v * v : d * (v - 0.5 * d) } / diff_data.size.to_f

        out = Tensor.create([loss_val], [1], requires_grad: pred.requires_grad || target.requires_grad)
        if pred.requires_grad || target.requires_grad
          out._grafo_hijos.push(pred, target)
          n = diff_data.size.to_f
          out.backward_fn = ->(g) {
            grad_pred = (pred - target).to_a.map do |err|
              abs_err = err.abs
              (abs_err <= d ? err : d * (err > 0 ? 1.0 : -1.0)) * (g.item / n)
            end
            pred.agregar_gradiente(Tensor.create(grad_pred, pred.shape)) if pred.requires_grad
            target.agregar_gradiente(Tensor.create(grad_pred.map(&:-@), target.shape)) if target.requires_grad
          }
        end
        out
      end
    end
  end
end
