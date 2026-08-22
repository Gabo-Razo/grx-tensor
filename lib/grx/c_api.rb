# frozen_string_literal: true

require "fiddle"
require "fiddle/import"

module GRX
  module CAPI
    extend Fiddle::Importer

    CANDIDATE_NAMES = case RUBY_PLATFORM
                      when /mingw|mswin|windows/i
                        ["grx_core.dll", "libgrx_core.dll", "libgrx_core.so", "grx_core.so"]
                      when /darwin/i
                        ["libgrx_core.dylib", "grx_core.bundle", "libgrx_core.so", "grx_core.so"]
                      else
                        ["libgrx_core.so", "grx_core.so", "libgrx_core.dylib", "grx_core.dll"]
                      end

    SEARCH_DIRS = [
      File.expand_path(__dir__),                     # lib/grx/
      File.expand_path("..", __dir__),              # lib/
      File.expand_path("../../ext/grx", __dir__),    # ext/grx/
      File.expand_path("../../ext/unix", __dir__),   # ext/unix/
      File.expand_path("../../ext/windows", __dir__) # ext/windows/
    ].freeze

    LIB_PATHS = SEARCH_DIRS.flat_map do |dir|
      CANDIDATE_NAMES.flat_map do |name|
        [
          File.join(dir, name),
          File.join(dir, name).tr("/", "\\")
        ]
      end
    end.uniq.freeze

    LOADED = begin
      path = LIB_PATHS.find { |p| File.file?(p) && File.exist?(p) }
      if path
        dlload path
        true
      else
        raise Fiddle::DLError, "Binary library not found (#{CANDIDATE_NAMES.join(', ')}) in #{SEARCH_DIRS.inspect}"
      end
    rescue Fiddle::DLError => e
      warn "[GRX] C extension unavailable: #{e.message}\n" \
           "      → Run: make -C ext/unix all (Linux/macOS) or make -C ext/windows -f Makefile.mingw all (Windows)\n" \
           "      → Running in pure Ruby fallback mode (without SIMD)."
      false
    end

    if LOADED
      # Memory management
      extern "double* grx_alloc(size_t)"
      extern "void    grx_free(double*)"

      # Element-wise arithmetic
      extern "void grx_add       (double*, double*, double*, size_t)"
      extern "void grx_sub       (double*, double*, double*, size_t)"
      extern "void grx_mul       (double*, double*, double*, size_t)"
      extern "void grx_div       (double*, double*, double*, size_t)"
      extern "void grx_scale     (double*, double,  double*, size_t)"
      extern "void grx_add_scalar(double*, double,  double*, size_t)"
      extern "void grx_negate    (double*, double*, size_t)"

      # Element-wise math
      extern "void grx_abs   (double*, double*, size_t)"
      extern "void grx_sqrt  (double*, double*, size_t)"
      extern "void grx_square(double*, double*, size_t)"
      extern "void grx_log   (double*, double*, size_t)"
      extern "void grx_exp   (double*, double*, size_t)"
      extern "void grx_pow   (double*, double,  double*, size_t)"
      extern "void grx_clip  (double*, double, double, double*, size_t)"

      # Reductions
      extern "double grx_sum (double*, size_t)"
      extern "double grx_mean(double*, size_t)"
      extern "double grx_max (double*, size_t)"
      extern "double grx_min (double*, size_t)"

      # Linear algebra
      extern "double grx_dot    (double*, double*, size_t)"
      extern "void   grx_matmul (double*, double*, double*, size_t, size_t, size_t)"

      # Activations
      extern "void grx_relu       (double*, double*, size_t)"
      extern "void grx_leaky_relu (double*, double, double*, size_t)"
      extern "void grx_tanh_act   (double*, double*, size_t)"
      extern "void grx_sigmoid    (double*, double*, size_t)"
      extern "void grx_softmax    (double*, double*, size_t)"

      # Optimizers
      extern "void grx_sgd_step (double*, double*, double, size_t)"
      extern "void grx_adam_step(double*, double*, double*, double*, double, double, double, double, double, double, size_t)"

      # Weight initialization
      extern "void grx_init_xavier_uniform(double*, size_t, size_t, size_t)"
      extern "void grx_init_he_normal     (double*, size_t, size_t)"
    end
  end
end
