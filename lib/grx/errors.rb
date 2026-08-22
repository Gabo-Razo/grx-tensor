# frozen_string_literal: true

module GRX
  class Error          < StandardError; end
  class ShapeError     < Error; end
  class DimensionError < Error; end
  class StorageError   < Error; end
end
