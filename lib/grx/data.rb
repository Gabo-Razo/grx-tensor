# frozen_string_literal: true

module GRX
  module Data
    # ================================================================
    # Dataset — Base class for data collections
    # ================================================================
    class Dataset
      def size
        raise NotImplementedError, "#{self.class}#size must be implemented"
      end

      def [](index)
        raise NotImplementedError, "#{self.class}#[] must be implemented"
      end
    end

    # ================================================================
    # TensorDataset — Dataset wrapping parallel tensors (e.g. X and Y)
    # ================================================================
    class TensorDataset < Dataset
      attr_reader :tensors, :size

      def initialize(*tensors)
        raise ArgumentError, "Must provide at least one tensor" if tensors.empty?
        first_dim = tensors.first.shape[0]
        unless tensors.all? { |t| t.shape[0] == first_dim }
          raise ArgumentError, "All tensors must have the same size in batch dimension (dimension 0)"
        end
        @tensors = tensors
        @size = first_dim
      end

      def [](index)
        @tensors.map do |t|
          cols = t.numel / @size
          offset = index * cols
          data = t.to_a.slice(offset, cols)
          new_shape = t.shape.size == 1 ? [1] : [1] + t.shape[1..]
          Tensor.create(data, new_shape)
        end
      end
    end

    # ================================================================
    # DataLoader — Mini-batch iterator with optional shuffling
    # ================================================================
    class DataLoader
      include Enumerable

      attr_reader :dataset, :batch_size, :shuffle

      def initialize(dataset, batch_size: 32, shuffle: true)
        @dataset    = dataset
        @batch_size = batch_size
        @shuffle    = shuffle
      end

      def each
        return to_enum(:each) unless block_given?

        indices = (0...@dataset.size).to_a
        indices.shuffle! if @shuffle

        indices.each_slice(@batch_size) do |batch_indices|
          batch_samples = batch_indices.map { |i| @dataset[i] }
          num_tensors = batch_samples.first.size

          batched = (0...num_tensors).map do |t_idx|
            slices = batch_samples.map { |sample| sample[t_idx].to_a }
            flat_data = slices.flatten
            sample_shape = batch_samples.first[t_idx].shape
            batch_dim = batch_indices.size
            target_shape = [batch_dim] + sample_shape[1..]
            Tensor.create(flat_data, target_shape)
          end

          yield(*batched)
        end
      end

      def size
        (@dataset.size.to_f / @batch_size).ceil
      end
    end
  end
end
