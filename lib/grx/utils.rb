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
  end
end
