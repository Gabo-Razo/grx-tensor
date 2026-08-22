# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/grx"

EPSILON = 1e-9

class TestTensor < Minitest::Test

  # ------------------------------------------------------------------
  # Factories
  # ------------------------------------------------------------------
  def test_create_y_get
    t = GRX::Tensor.create([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert_in_delta 1.0, t.get(0, 0), EPSILON
    assert_in_delta 2.0, t.get(0, 1), EPSILON
    assert_in_delta 3.0, t.get(1, 0), EPSILON
    assert_in_delta 4.0, t.get(1, 1), EPSILON
  end

  def test_zeros
    t = GRX::Tensor.zeros([3])
    assert_equal [0.0, 0.0, 0.0], t.to_a
  end

  def test_ones
    t = GRX::Tensor.ones([3])
    assert_equal [1.0, 1.0, 1.0], t.to_a
  end

  # ------------------------------------------------------------------
  # Operaciones element-wise
  # ------------------------------------------------------------------
  def test_suma
    a = GRX.tensor([1.0, 2.0, 3.0], [3])
    b = GRX.tensor([4.0, 5.0, 6.0], [3])
    assert_array_close [5.0, 7.0, 9.0], (a + b).to_a
  end

  def test_resta
    a = GRX.tensor([10.0, 20.0], [2])
    b = GRX.tensor([3.0,  5.0],  [2])
    assert_array_close [7.0, 15.0], (a - b).to_a
  end

  def test_multiplicacion
    a = GRX.tensor([2.0, 3.0], [2])
    b = GRX.tensor([4.0, 5.0], [2])
    assert_array_close [8.0, 15.0], (a * b).to_a
  end

  def test_scale
    a = GRX.tensor([1.0, 2.0, 3.0], [3])
    assert_array_close [2.5, 5.0, 7.5], a.scale(2.5).to_a
  end

  def test_negate
    a = GRX.tensor([1.0, -2.0, 3.0], [3])
    assert_array_close [-1.0, 2.0, -3.0], a.negate.to_a
  end

  # ------------------------------------------------------------------
  # Álgebra lineal
  # ------------------------------------------------------------------
  def test_dot
    a = GRX.tensor([1.0, 2.0, 3.0], [3])
    b = GRX.tensor([4.0, 5.0, 6.0], [3])
    assert_in_delta 32.0, a.dot(b), EPSILON  # 1*4 + 2*5 + 3*6 = 32
  end

  def test_matmul
    # [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
    a = GRX.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = GRX.tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = a.matmul(b)
    assert_equal [2, 2], c.shape
    assert_array_close [19.0, 22.0, 43.0, 50.0], c.to_a
  end

  # ------------------------------------------------------------------
  # Activaciones
  # ------------------------------------------------------------------
  def test_relu
    a = GRX.tensor([-2.0, 0.0, 3.0, -0.5], [4])
    assert_array_close [0.0, 0.0, 3.0, 0.0], a.relu.to_a
  end

  def test_sigmoid
    a = GRX.tensor([0.0], [1])
    assert_in_delta 0.5, a.sigmoid.to_a[0], 1e-6
  end

  def test_softmax_suma_uno
    a = GRX.tensor([1.0, 2.0, 3.0], [3])
    s = a.softmax.to_a
    assert_in_delta 1.0, s.sum, 1e-9
    s.each { |v| assert v >= 0.0 }
  end

  # ------------------------------------------------------------------
  # Geometría
  # ------------------------------------------------------------------
  def test_reshape
    t = GRX.tensor([1.0, 2.0, 3.0, 4.0], [4])
    m = t.reshape([2, 2])
    assert_equal [2, 2], m.shape
    assert_in_delta 3.0, m.get(1, 0), EPSILON
  end

  def test_transpose
    t = GRX.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    tr = t.transpose
    assert_equal [2, 2], tr.shape
    assert_in_delta 3.0, tr.get(0, 1), EPSILON
    assert_in_delta 2.0, tr.get(1, 0), EPSILON
  end

  # ------------------------------------------------------------------
  # Autograd
  # ------------------------------------------------------------------
  def test_autograd_suma
    a = GRX.tensor([2.0, 3.0], [2], requires_grad: true)
    b = GRX.tensor([4.0, 5.0], [2], requires_grad: true)
    c = a + b
    assert c.requires_grad
    c.backward
    assert_array_close [1.0, 1.0], a.grad.to_a
    assert_array_close [1.0, 1.0], b.grad.to_a
  end

  def test_autograd_mul
    a = GRX.tensor([2.0, 3.0], [2], requires_grad: true)
    b = GRX.tensor([4.0, 5.0], [2], requires_grad: true)
    c = a * b
    c.backward
    # d(a*b)/da = b
    assert_array_close [4.0, 5.0], a.grad.to_a
    assert_array_close [2.0, 3.0], b.grad.to_a
  end

  def test_autograd_cadena
    # c = (a + b) * b  →  dc/da = b,  dc/db = a + 2b
    a = GRX.tensor([1.0, 2.0], [2], requires_grad: true)
    b = GRX.tensor([3.0, 4.0], [2], requires_grad: true)
    c = (a + b) * b
    c.backward
    assert_array_close [3.0, 4.0], a.grad.to_a          # dc/da = b
    assert_array_close [7.0, 10.0], b.grad.to_a         # dc/db = a + 2b = [1+6, 2+8]
  end

  # ------------------------------------------------------------------
  # Errores
  # ------------------------------------------------------------------
  def test_shapes_incompatibles
    a = GRX.tensor([1.0, 2.0], [2])
    b = GRX.tensor([1.0, 2.0, 3.0], [3])
    assert_raises(GRX::ShapeError) { a + b }
  end

  def test_reshape_incompatible
    t = GRX.tensor([1.0, 2.0, 3.0], [3])
    assert_raises(ArgumentError) { t.reshape([2, 2]) }
  end

  # ------------------------------------------------------------------
  # Factory shorthand
  # ------------------------------------------------------------------
  def test_factory_shorthand
    t = GRX.tensor([1.0, 2.0, 3.0], [3])
    assert_equal [3], t.shape
    assert_in_delta 2.0, t.get(1), EPSILON
  end

  private

  def assert_array_close(expected, actual, delta = EPSILON)
    assert_equal expected.size, actual.size, "Arrays de distinto tamaño"
    expected.zip(actual).each_with_index do |(e, a), i|
      assert_in_delta e, a, delta, "Diferencia en índice #{i}: esperado #{e}, obtenido #{a}"
    end
  end
end
