# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/grx"

class TestNN < Minitest::Test

  # ----------------------------------------------------------------
  # Operaciones nuevas
  # ----------------------------------------------------------------
  def test_div
    a = GRX.tensor([10.0, 6.0, 9.0], [3])
    b = GRX.tensor([2.0,  3.0, 3.0], [3])
    assert_array_close [5.0, 2.0, 3.0], (a / b).to_a
  end

  def test_div_scalar
    a = GRX.tensor([4.0, 8.0], [2])
    assert_array_close [2.0, 4.0], (a / 2.0).to_a
  end

  def test_abs
    a = GRX.tensor([-3.0, 0.0, 4.0], [3])
    assert_array_close [3.0, 0.0, 4.0], a.abs.to_a
  end

  def test_sqrt
    a = GRX.tensor([4.0, 9.0, 16.0], [3])
    assert_array_close [2.0, 3.0, 4.0], a.sqrt.to_a
  end

  def test_square
    a = GRX.tensor([2.0, 3.0, 4.0], [3])
    assert_array_close [4.0, 9.0, 16.0], a.square.to_a
  end

  def test_exp_log
    a = GRX.tensor([0.0, 1.0], [2])
    exp_a = a.exp
    assert_in_delta 1.0,        exp_a.to_a[0], 1e-9
    assert_in_delta Math::E,    exp_a.to_a[1], 1e-9
    assert_in_delta 0.0,        exp_a.log.to_a[0], 1e-9
    assert_in_delta 1.0,        exp_a.log.to_a[1], 1e-9
  end

  def test_pow
    a = GRX.tensor([2.0, 3.0], [2])
    assert_array_close [8.0, 27.0], a.pow(3).to_a
  end

  def test_clip
    a = GRX.tensor([-5.0, 0.5, 3.0, 10.0], [4])
    assert_array_close [0.0, 0.5, 1.0, 1.0], a.clip(0.0, 1.0).to_a
  end

  # ----------------------------------------------------------------
  # Reducciones
  # ----------------------------------------------------------------
  def test_sum
    a = GRX.tensor([1.0, 2.0, 3.0, 4.0], [4])
    assert_in_delta 10.0, a.sum, 1e-9
  end

  def test_mean
    a = GRX.tensor([2.0, 4.0, 6.0], [3])
    assert_in_delta 4.0, a.mean, 1e-9
  end

  def test_max_min
    a = GRX.tensor([3.0, 1.0, 4.0, 1.0, 5.0], [5])
    assert_in_delta 5.0, a.max, 1e-9
    assert_in_delta 1.0, a.min, 1e-9
  end

  # ----------------------------------------------------------------
  # Activaciones
  # ----------------------------------------------------------------
  def test_tanh
    a = GRX.tensor([0.0], [1])
    assert_in_delta 0.0, a.tanh.to_a[0], 1e-9
    b = GRX.tensor([100.0], [1])
    assert_in_delta 1.0, b.tanh.to_a[0], 1e-6
  end

  def test_leaky_relu
    a = GRX.tensor([-2.0, 0.0, 3.0], [3])
    r = a.leaky_relu(0.1)
    assert_in_delta(-0.2, r.to_a[0], 1e-9)
    assert_in_delta 0.0,  r.to_a[1], 1e-9
    assert_in_delta 3.0,  r.to_a[2], 1e-9
  end

  # ----------------------------------------------------------------
  # Capas NN
  # ----------------------------------------------------------------
  def test_linear_shape
    layer = GRX::NN::Linear.new(4, 3)
    x = GRX.tensor(Array.new(8, 1.0), [2, 4])  # batch=2, in=4
    out = layer.call(x)
    assert_equal [2, 3], out.shape
  end

  def test_sequential
    net = GRX::NN::Sequential.new(
      GRX::NN::Linear.new(4, 8),
      GRX::NN::ReLU.new,
      GRX::NN::Linear.new(8, 2)
    )
    x = GRX.tensor(Array.new(4) { rand }, [1, 4])
    out = net.call(x)
    assert_equal [1, 2], out.shape
  end

  def test_parameters
    net = GRX::NN::Sequential.new(
      GRX::NN::Linear.new(3, 4),
      GRX::NN::Linear.new(4, 2)
    )
    # Linear(3,4): weight[4,3] + bias[4] = 16 params
    # Linear(4,2): weight[2,4] + bias[2] = 10 params
    assert_equal 4, net.parameters.size  # 2 weights + 2 biases
  end

  # ----------------------------------------------------------------
  # Loss functions
  # ----------------------------------------------------------------
  def test_mse_loss
    pred   = GRX.tensor([1.0, 2.0, 3.0], [3])
    target = GRX.tensor([1.0, 2.0, 3.0], [3])
    assert_in_delta 0.0, GRX::Loss::MSELoss.new.call(pred, target), 1e-9

    pred2  = GRX.tensor([0.0, 0.0, 0.0], [3])
    # MSE = mean([1,4,9]) = 14/3
    assert_in_delta 14.0/3, GRX::Loss::MSELoss.new.call(pred2, target), 1e-9
  end

  def test_bce_loss_perfect
    pred   = GRX.tensor([0.9999, 0.0001], [2])
    target = GRX.tensor([1.0,    0.0],    [2])
    loss = GRX::Loss::BCELoss.new.call(pred, target)
    assert loss < 0.01, "BCE con predicción perfecta debe ser ~0, fue #{loss}"
  end

  # ----------------------------------------------------------------
  # Optimizadores — verificamos que el loss baja
  # ----------------------------------------------------------------
  def test_sgd_reduces_loss
    # Regresión lineal simple: y = 2x, aprendemos el peso
    w = GRX.tensor([0.0], [1], requires_grad: true)
    opt = GRX::Optim::SGD.new([w], lr: 0.1)
    loss_fn = GRX::Loss::MSELoss.new

    x = GRX.tensor([1.0], [1])
    y = GRX.tensor([2.0], [1])

    losses = []
    20.times do
      opt.zero_grad
      pred = w * x
      loss_val = loss_fn.call(pred, y)
      # Backprop manual para escalar
      diff = pred.to_a[0] - y.to_a[0]
      w.agregar_gradiente(GRX.tensor([2.0 * diff], [1]))
      opt.step
      losses << (w.to_a[0] - 2.0).abs
    end

    assert losses.last < losses.first, "SGD debe reducir el error"
    assert losses.last < 0.1, "SGD debe converger cerca de w=2.0"
  end

  def test_adam_reduces_loss
    w = GRX.tensor([0.0], [1], requires_grad: true)
    opt = GRX::Optim::Adam.new([w], lr: 0.3)

    initial_err = (w.to_a[0] - 2.0).abs
    50.times do
      opt.zero_grad
      diff = w.to_a[0] - 2.0
      w.agregar_gradiente(GRX.tensor([2.0 * diff], [1]))
      opt.step
    end
    final_err = (w.to_a[0] - 2.0).abs

    assert final_err < initial_err, "Adam debe reducir el error"
    assert final_err < 0.1, "Adam debe converger, error final: #{final_err}"
  end

  # ----------------------------------------------------------------
  # Inicialización de pesos
  # ----------------------------------------------------------------
  def test_xavier_uniform_range
    t = GRX::Tensor.xavier_uniform([100, 100])
    limit = Math.sqrt(6.0 / 200)
    t.to_a.each { |v| assert v.abs <= limit + 1e-9, "Xavier fuera de rango: #{v}" }
  end

  def test_he_normal_stats
    t = GRX::Tensor.he_normal([1000, 100])
    # Media debe ser ~0, std ~sqrt(2/100) = ~0.141
    vals = t.to_a
    mean = vals.sum / vals.size
    std  = Math.sqrt(vals.sum { |v| (v - mean)**2 } / vals.size)
    assert mean.abs < 0.05,       "He normal: media debe ser ~0, fue #{mean}"
    assert (std - 0.141).abs < 0.02, "He normal: std debe ser ~0.141, fue #{std}"
  end

  # ----------------------------------------------------------------
  # Capas avanzadas: Embedding y LayerNorm
  # ----------------------------------------------------------------
  def test_embedding
    emb = GRX::NN::Embedding.new(20, 8)
    assert_equal [20, 8], emb.weight.shape
    tokens = GRX.tensor([2, 5, 19], [3])
    out = emb.call(tokens)
    assert_equal [3, 8], out.shape
    out.backward
    refute_nil emb.weight.grad
    assert_equal [20, 8], emb.weight.grad.shape
  end

  def test_layer_norm
    ln = GRX::NN::LayerNorm.new(4)
    x = GRX.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], [2, 4], requires_grad: true)
    out = ln.call(x)
    assert_equal [2, 4], out.shape
    # Cada fila normalizada debe tener media ~0 y varianza ~1
    row0 = out.to_a[0...4]
    assert_in_delta 0.0, row0.sum / 4.0, 1e-4
    out.backward
    refute_nil x.grad
    assert_equal [2, 4], x.grad.shape
  end

  # ----------------------------------------------------------------
  # Serialización .grx
  # ----------------------------------------------------------------
  def test_save_and_load_weights_grx
    net1 = GRX::NN::Sequential.new(
      GRX::NN::Linear.new(3, 6),
      GRX::NN::Tanh.new,
      GRX::NN::Linear.new(6, 2)
    )
    x = GRX.tensor([1.0, 2.0, 3.0], [1, 3])
    pred1 = net1.call(x).to_a

    tmp_file = "/tmp/test_suite_model.grx"
    net1.save_weights(tmp_file)
    assert File.exist?(tmp_file)
    assert File.size(tmp_file) > 0

    net2 = GRX::NN::Sequential.new(
      GRX::NN::Linear.new(3, 6),
      GRX::NN::Tanh.new,
      GRX::NN::Linear.new(6, 2)
    )
    net2.load_weights(tmp_file)
    pred2 = net2.call(x).to_a

    assert_array_close pred1, pred2, 1e-12
    File.delete(tmp_file) if File.exist?(tmp_file)
  end

  # ----------------------------------------------------------------
  # Data: TensorDataset & DataLoader
  # ----------------------------------------------------------------
  def test_dataloader_batching_and_size
    x = GRX.tensor((1..20).map(&:to_f), [20, 1])
    y = GRX.tensor((1..20).map { |v| 2.0 * v }, [20, 1])
    dataset = GRX::Data::TensorDataset.new(x, y)
    assert_equal 20, dataset.size

    loader = GRX::Data::DataLoader.new(dataset, batch_size: 6, shuffle: false)
    assert_equal 4, loader.size # 6 + 6 + 6 + 2 = 20 (4 lotes)

    batches = loader.to_a
    assert_equal 4, batches.size
    assert_equal [6, 1], batches[0][0].shape
    assert_equal [2, 1], batches[3][0].shape
  end

  # ----------------------------------------------------------------
  # Utils: Gradient Clipping
  # ----------------------------------------------------------------
  def test_clip_grad_norm
    p1 = GRX.tensor([3.0, 4.0], [2], requires_grad: true)
    p1.agregar_gradiente(GRX.tensor([3.0, 4.0], [2])) # norma = 5.0
    norm = GRX::Utils.clip_grad_norm!([p1], 2.5)
    assert_in_delta 5.0, norm, 1e-6
    # Gradientes escalados a la mitad
    assert_array_close [1.5, 2.0], p1.grad.to_a, 1e-4
  end

  # ----------------------------------------------------------------
  # Loss.backward directo
  # ----------------------------------------------------------------
  def test_loss_backward_direct
    w = GRX.tensor([0.0], [1], requires_grad: true)
    x = GRX.tensor([1.0], [1])
    y = GRX.tensor([2.0], [1])
    loss_fn = GRX::Loss::MSELoss.new
    opt = GRX::Optim::SGD.new([w], lr: 0.1)

    opt.zero_grad
    pred = w * x
    loss = loss_fn.call(pred, y)
    loss.backward
    opt.step
    # Gradiente de (0 - 2)^2 = 2 * (-2) = -4.0
    # w nuevo = 0 - 0.1 * (-4.0) = 0.4
    assert_in_delta 0.4, w.to_a[0], 1e-6
  end

  private

  def assert_array_close(expected, actual, delta = 1e-9)
    assert_equal expected.size, actual.size
    expected.zip(actual).each_with_index do |(e, a), i|
      assert_in_delta e, a, delta, "índice #{i}: esperado #{e}, obtenido #{a}"
    end
  end
end
