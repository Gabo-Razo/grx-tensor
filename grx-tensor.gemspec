# frozen_string_literal: true

require_relative "lib/grx/version"

Gem::Specification.new do |spec|
  spec.name          = "grx-tensor"
  spec.version       = GRX::VERSION
  spec.authors       = ["Razo"]
  spec.email         = ["garabatoangelopolis@gmail.com"]

  spec.summary       = "Tensor framework for Ruby with autograd and a C+SIMD compute core"
  spec.description   = <<~DESC
    GRX brings PyTorch-style tensor operations to Ruby. Every arithmetic op,
    activation, and optimizer step runs through a native C library compiled
    with AVX2+FMA SIMD. Ruby is the interface — C does the work.

    Features: autograd, SGD/Adam optimizers, Linear/Sequential/Dropout/BatchNorm
    layers, MSE/BCE/CrossEntropy loss functions, Xavier and He weight init.
    Cross-platform: .so on Linux, .dylib on macOS, .dll on Windows.
  DESC

  spec.homepage      = "https://github.com/Gabo-Razo/grx-tensor"
  spec.license       = "MIT"

  spec.required_ruby_version = ">= 3.0.0"

  spec.metadata = {
    "homepage_uri"    => spec.homepage,
    "changelog_uri"   => "#{spec.homepage}/blob/main/CHANGELOG.md",
    "bug_tracker_uri" => "#{spec.homepage}/issues"
  }

  spec.files = Dir[
    "lib/**/*.rb",
    "ext/grx/**/*.{c,h,rb}",
    "ext/unix/Makefile",
    "ext/windows/Makefile.mingw",
    "*.gemspec",
    "README.md",
    "README.es.md",
    "GUIA_PRINCIPIANTES.md",
    "LICENSE.txt",
    "CHANGELOG.md"
  ].reject { |f| f.match?(/\.(so|dll|dylib|bundle|a)$/) }

  spec.require_paths = ["lib"]

  # rake-compiler compiles ext/grx/extconf.rb on `gem install`
  spec.extensions = ["ext/grx/extconf.rb"]

  spec.post_install_message = <<~MSG

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      GRX-Tensor #{GRX::VERSION} installed
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Compile the C extension to enable AVX2+FMA SIMD:

      Linux / macOS:  make -C ext/unix
      Windows:        make -C ext/windows -f Makefile.mingw

    Without it, GRX runs in pure Ruby fallback mode (slower but correct).

    Quick start:

      require "grx"

      a = GRX.tensor([1.0, 2.0, 3.0], [3], requires_grad: true)
      b = GRX.tensor([4.0, 5.0, 6.0], [3], requires_grad: true)
      c = a + b
      c.backward
      puts a.grad.to_a   # [1.0, 1.0, 1.0]

      net = GRX::NN::Sequential.new(
        GRX::NN::Linear.new(4, 16),
        GRX::NN::ReLU.new,
        GRX::NN::Linear.new(16, 1)
      )
      opt = GRX::Optim::Adam.new(net.parameters, lr: 0.001)

    Docs: https://github.com/Gabo-Razo/grx-tensor
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MSG

  spec.add_development_dependency "rake",          "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.2"
  spec.add_development_dependency "minitest",      "~> 5.0"
  spec.add_development_dependency "bundler",       "~> 2.0"
end
