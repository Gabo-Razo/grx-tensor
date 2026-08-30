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
    activation, and optimizer step runs through a native C library with
    dynamic multi-target SIMD dispatch (AVX2+FMA, SSE, and scalar fallback).
    Ruby is the interface — C does the work.

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

  # rake-compiler / rubygems compiles ext/grx/extconf.rb on `gem install`
  spec.extensions = ["ext/grx/extconf.rb"]

  spec.post_install_message = <<~MSG

    ============================================================
      GRX-Tensor #{GRX::VERSION}
    ============================================================

      [ENGLISH]
      Thank you for choosing and using GRX-Tensor!

      * Native C acceleration with dynamic SIMD dispatch
        (AVX2+FMA, SSE, and portable scalar C) is built and
        configured automatically.
      * Windows Note: Native C acceleration is supported when
        using RubyInstaller with DevKit (MSYS2 / MinGW-w64).
        Standalone pre-compiled Windows binaries are currently
        in active development.
      * Documentation, guides, and tutorials:
        https://github.com/Gabo-Razo/grx-tensor

      ----------------------------------------------------------

      [ESPAÑOL]
      Muchas gracias por elegir y utilizar GRX-Tensor!

      * La aceleracion nativa en C con despacho dinamico SIMD
        (AVX2+FMA, SSE y C escalar portable) se compila y
        configura de forma totalmente automatica.
      * Nota para Windows: La aceleracion nativa esta soportada
        al utilizar RubyInstaller con DevKit (MSYS2 / MinGW-w64).
        Los binarios pre-compilados independientes para Windows
        se encuentran actualmente en desarrollo activo.
      * Documentacion, guias y tutoriales:
        https://github.com/Gabo-Razo/grx-tensor

    ============================================================

  MSG

  spec.add_development_dependency "rake",          "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.2"
  spec.add_development_dependency "minitest",      "~> 5.0"
  spec.add_development_dependency "bundler",       "~> 2.0"
end
