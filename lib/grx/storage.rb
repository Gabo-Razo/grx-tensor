# frozen_string_literal: true

require "fiddle"

module GRX
  # ===================================================================
  # Storage — Native memory buffer
  #
  # When CAPI is loaded:
  #   @ptr  → Fiddle::Pointer to 32-byte aligned doubles block
  #           allocated via grx_alloc() (C posix_memalign / _aligned_malloc).
  #           Data lives in C heap, NOT managed by Ruby GC.
  #
  # When CAPI is NOT loaded (fallback):
  #   @data → Standard Ruby Array (slow but correct).
  # ===================================================================
  class Storage
    attr_reader :size
    attr_reader :ptr

    def initialize(array_plano)
      @size = array_plano.size

      if CAPI::LOADED
        # Fast mode: aligned C memory
        @ptr = CAPI.grx_alloc(@size)
        raise StorageError, "grx_alloc failed (OOM)" if @ptr.null?

        # Pack Ruby Array into C buffer as IEEE 754 doubles
        bytes = array_plano.pack("d*")
        @ptr[0, bytes.bytesize] = bytes

        # Finalizer releases C memory upon Ruby GC collection
        ptr_to_free = @ptr
        ObjectSpace.define_finalizer(self, self.class.make_finalizer(ptr_to_free))
      else
        # Fallback mode: Ruby Array
        @data = array_plano.map(&:to_f)
        @ptr  = nil
      end
    end

    def self.make_finalizer(ptr)
      proc { CAPI.grx_free(ptr) }
    end

    # ------------------------------------------------------------------
    # Read / Write — used in fallback mode and by item/get()
    # High-performance tensor ops operate directly on @ptr in C.
    # ------------------------------------------------------------------
    def read(indice)
      if CAPI::LOADED
        @ptr[indice * 8, 8].unpack1("d")
      else
        @data[indice]
      end
    end

    def write(indice, valor)
      if CAPI::LOADED
        @ptr[indice * 8, 8] = [valor.to_f].pack("d")
      else
        @data[indice] = valor.to_f
      end
    end

    # Dumps entire buffer to a Ruby Array
    def to_ruby_array
      if CAPI::LOADED
        @ptr[0, @size * 8].unpack("d#{@size}")
      else
        @data.dup
      end
    end
  end
end
