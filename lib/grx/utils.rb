# frozen_string_literal: true

module GRX
  module Utils
    # ================================================================
    # clip_grad_norm! — Clips gradients of parameter collection so their
    # combined L2 norm does not exceed max_norm.
    # Prevents exploding gradient problems during deep network training.
    # ================================================================
    def self.clip_grad_norm!(parameters, max_norm)
      max_norm = max_norm.to_f
      total_norm_sq = 0.0
      parameters.each do |p|
        next unless p.grad
        total_norm_sq += p.grad.square.to_a.sum
      end
      total_norm = Math.sqrt(total_norm_sq)
      clip_coef = max_norm / (total_norm + 1e-6)
      if clip_coef < 1.0
        parameters.each do |p|
          next unless p.grad
          p.grad = p.grad.scale(clip_coef)
        end
      end
      total_norm
    end

    # ================================================================
    # one_hot — Generates a 2D One-Hot encoded Tensor from class indices
    # ================================================================
    def self.one_hot(indices, num_classes: nil, requires_grad: false)
      ids = indices.is_a?(Tensor) ? indices.to_a.map(&:to_i) : Array(indices).map(&:to_i)
      c = num_classes || (ids.empty? ? 0 : ids.max + 1)
      n = ids.size
      matrix_data = Array.new(n * c, 0.0)
      ids.each_with_index do |class_id, row|
        raise IndexError, "Class index #{class_id} out of bounds [0, #{c})" if class_id < 0 || class_id >= c
        matrix_data[row * c + class_id] = 1.0
      end
      Tensor.create(matrix_data, [n, c], requires_grad: requires_grad)
    end
  end
end
