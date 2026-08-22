# frozen_string_literal: true

module GRX
  # ===================================================================
  # Serialization — Native binary .grx format
  #
  # Binary layout:
  # - 8 bytes:  Magic header "GRX1\0\0\0\0"
  # - 4 bytes:  Number of parameter tensors (unsigned 32-bit big-endian)
  # For each tensor:
  #   - 2 bytes: Rank (number of dimensions)
  #   - 4 bytes * rank: Dimensions of the shape (uint32 big-endian)
  #   - 8 bytes: Total numel (uint64 big-endian)
  #   - numel * 8 bytes: Direct binary packed IEEE 754 doubles
  # ===================================================================
  module Serialization
    MAGIC = "GRX1\x00\x00\x00\x00".b

    def self.save(model, path)
      params = model.parameters
      File.open(path, "wb") do |f|
        f.write(MAGIC)
        f.write([params.size].pack("N"))
        params.each do |p|
          shape = p.shape
          f.write([shape.size].pack("n"))
          f.write(shape.pack("N*"))
          f.write([p.numel].pack("Q>"))
          # Direct binary copy from native C buffer
          bytes = if p.storage.ptr
            p.storage.ptr[0, p.numel * 8]
          else
            p.to_a.pack("d*")
          end
          f.write(bytes)
        end
      end
      path
    end

    def self.load(model, path)
      params = model.parameters
      File.open(path, "rb") do |f|
        magic = f.read(8)
        raise StorageError, "Invalid format: not a valid .grx binary file" unless magic == MAGIC
        count = f.read(4).unpack1("N")
        raise StorageError, "Parameter count mismatch: model has #{params.size}, file has #{count}" unless count == params.size

        params.each_with_index do |p, idx|
          rank = f.read(2).unpack1("n")
          shape = f.read(rank * 4).unpack("N*")
          numel = f.read(8).unpack1("Q>")
          raise ShapeError, "Shape mismatch for parameter #{idx}: expected #{p.shape}, got #{shape}" unless shape == p.shape

          bytes = f.read(numel * 8)
          if p.storage.ptr
            p.storage.ptr[0, bytes.bytesize] = bytes
          else
            p.storage.instance_variable_set(:@data, bytes.unpack("d*"))
          end
        end
      end
      model
    end
  end
end
