# frozen_string_literal: true

# =====================================================================
# GRX — Tensor framework with autograd and C+SIMD compute core
# require "grx"
# =====================================================================

require_relative "grx/version"
require_relative "grx/errors"
require_relative "grx/c_api"
require_relative "grx/storage"
require_relative "grx/tensor"
require_relative "grx/serialization"
require_relative "grx/nn"
require_relative "grx/optim"
require_relative "grx/loss"
require_relative "grx/data"
require_relative "grx/utils"

module GRX
  # Quick factory helpers
  def self.tensor(data, shape, requires_grad: false)
    Tensor.create(data, shape, requires_grad: requires_grad)
  end

  def self.zeros(shape, requires_grad: false)
    Tensor.zeros(shape, requires_grad: requires_grad)
  end

  def self.ones(shape, requires_grad: false)
    Tensor.ones(shape, requires_grad: requires_grad)
  end

  def self.rand(shape, requires_grad: false)
    n = shape.reduce(1, :*)
    Tensor.create(Array.new(n) { ::Kernel.rand }, shape, requires_grad: requires_grad)
  end

  def self.randn(shape, requires_grad: false)
    # Box-Muller from Ruby (C backend executes faster via he_normal)
    n = shape.reduce(1, :*)
    data = []
    (n / 2.0).ceil.times do
      u1 = ::Kernel.rand; u1 = ::Kernel.rand while u1 < 1e-15
      u2 = ::Kernel.rand
      r  = Math.sqrt(-2.0 * Math.log(u1))
      data << r * Math.cos(2 * Math::PI * u2)
      data << r * Math.sin(2 * Math::PI * u2)
    end
    Tensor.create(data.first(n), shape, requires_grad: requires_grad)
  end

  def self.c_loaded?
    CAPI::LOADED
  end

  def self.mode
    CAPI::LOADED ? :c : :ruby
  end
end
