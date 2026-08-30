# frozen_string_literal: true
# =============================================================
# GRX-Tensor — Full integration test
# Tests every feature the framework supports, end to end.
# Run with: ruby test/test_full.rb  (no -Ilib, uses installed gem)
# =============================================================

require "grx"

PASS = []; FAIL = []
EPS  = 1e-6

def assert_equal_f(desc, expected, actual, delta = EPS)
  ok = case expected
       when Array   then expected.size == actual.size &&
                         expected.zip(actual).all? { |e,a| (e-a).abs < delta }
       when true, false then expected == actual
       else (expected - actual).abs < delta
       end
  if ok then PASS << desc; print "  \e[32m✓\e[0m #{desc}\n"
  else       FAIL << desc; print "  \e[31m✗\e[0m #{desc}  expected=#{expected.inspect} got=#{actual.inspect}\n"
  end
end

def section(name)
  puts "\n\e[1m[ #{name} ]\e[0m"
end

puts "\e[1m" + "=" * 60 + "\e[0m"
puts "\e[1mGRX-Tensor #{GRX::VERSION} — Full Test Suite\e[0m"
puts "Mode: #{GRX::CAPI::LOADED ? "\e[32mC + SIMD ⚡\e[0m" : "\e[33mRuby fallback\e[0m"}"
puts "=" * 60

# ================================================================
section "Tensor creation & factories"
# ================================================================
t = GRX.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
assert_equal_f "tensor shape",        [2, 2], t.shape
assert_equal_f "tensor rank",         2,      t.rank
assert_equal_f "tensor numel",        4,      t.numel
assert_equal_f "zeros",               [0.0, 0.0, 0.0], GRX.zeros([3]).to_a
assert_equal_f "ones",                [1.0, 1.0, 1.0], GRX.ones([3]).to_a
assert_equal_f "zeros_like shape",    [2,2], GRX::Tensor.zeros_like(t).shape
assert_equal_f "ones_like values",    [1.0,1.0,1.0,1.0], GRX::Tensor.ones_like(t).to_a
assert_equal_f "rand shape",          [4], GRX.rand([4]).shape
assert_equal_f "rand in [0,1)",       true, GRX.rand([100]).to_a.all? { |v| v >= 0 && v < 1 }
assert_equal_f "randn shape",         [4], GRX.randn([4]).shape
assert_equal_f "item",                5.0, GRX.tensor([5.0], [1]).item
t_mut = GRX.tensor([1.0, 2.0], [2])
t_mut.set(1, 99.0)
assert_equal_f "set method",          99.0, t_mut.get(1)

# ================================================================
section "Arithmetic — element-wise"
# ================================================================
a = GRX.tensor([1.0, 2.0, 3.0, 4.0], [4])
b = GRX.tensor([4.0, 3.0, 2.0, 1.0], [4])

assert_equal_f "add",         [5.0, 5.0, 5.0, 5.0], (a + b).to_a
assert_equal_f "sub",         [-3.0, -1.0, 1.0, 3.0], (a - b).to_a
assert_equal_f "mul",         [4.0, 6.0, 6.0, 4.0], (a * b).to_a
assert_equal_f "div",         [0.25, 2.0/3, 1.5, 4.0], (a / b).to_a
assert_equal_f "scale",       [2.0, 4.0, 6.0, 8.0], a.scale(2.0).to_a
assert_equal_f "add_scalar",  [11.0, 12.0, 13.0, 14.0], a.add_scalar(10.0).to_a
assert_equal_f "negate",      [-1.0, -2.0, -3.0, -4.0], a.negate.to_a
assert_equal_f "unary minus", [-1.0, -2.0, -3.0, -4.0], (-a).to_a
assert_equal_f "tensor + scalar", [2.0, 3.0, 4.0, 5.0], (a + 1.0).to_a
assert_equal_f "tensor - scalar", [0.0, 1.0, 2.0, 3.0], (a - 1.0).to_a
assert_equal_f "tensor * scalar", [3.0, 6.0, 9.0, 12.0], (a * 3.0).to_a
assert_equal_f "tensor / scalar", [0.5, 1.0, 1.5, 2.0], (a / 2.0).to_a

# ================================================================
section "Math ops — element-wise"
# ================================================================
x = GRX.tensor([1.0, 4.0, 9.0, 16.0], [4])
assert_equal_f "sqrt",    [1.0, 2.0, 3.0, 4.0], x.sqrt.to_a
assert_equal_f "square",  [1.0, 16.0, 81.0, 256.0], x.square.to_a
assert_equal_f "abs",     [1.0, 2.0, 3.0, 4.0], GRX.tensor([-1.0,-2.0,3.0,4.0],[4]).abs.to_a
assert_equal_f "pow(2)",  [1.0, 4.0, 9.0, 16.0], a.pow(2).to_a
assert_equal_f "pow(3)",  [1.0, 8.0, 27.0, 64.0], a.pow(3).to_a
assert_equal_f "exp/log", [1.0, 2.0, 3.0, 4.0], a.log.exp.to_a, 1e-5
assert_equal_f "clip lo", [2.0, 2.0, 3.0, 4.0], a.clip(2.0, 10.0).to_a
assert_equal_f "clip hi", [1.0, 2.0, 3.0, 3.0], a.clip(0.0, 3.0).to_a
assert_equal_f "clip both",[2.0, 2.0, 3.0, 3.0], a.clip(2.0, 3.0).to_a

# ================================================================
section "Reductions"
# ================================================================
r = GRX.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], [8])
assert_equal_f "sum",  31.0,    r.sum
assert_equal_f "mean", 31.0/8,  r.mean
assert_equal_f "max",  9.0,     r.max
assert_equal_f "min",  1.0,     r.min

# ================================================================
section "Linear algebra"
# ================================================================
u = GRX.tensor([1.0, 2.0, 3.0], [3])
v = GRX.tensor([4.0, 5.0, 6.0], [3])
assert_equal_f "dot",  32.0, u.dot(v)

# matmul 2×2
a2 = GRX.tensor([1.0,2.0,3.0,4.0], [2,2])
b2 = GRX.tensor([5.0,6.0,7.0,8.0], [2,2])
assert_equal_f "matmul 2x2", [19.0,22.0,43.0,50.0], a2.matmul(b2).to_a

# matmul 2×3 × 3×2
a3 = GRX.tensor([1.0,2.0,3.0, 4.0,5.0,6.0], [2,3])
b3 = GRX.tensor([7.0,8.0, 9.0,10.0, 11.0,12.0], [3,2])
c3 = a3.matmul(b3)
assert_equal_f "matmul 2x3 × 3x2 shape", [2,2], c3.shape
assert_equal_f "matmul 2x3 × 3x2 vals",  [58.0,64.0,139.0,154.0], c3.to_a

# ================================================================
section "Geometry — zero-copy views"
# ================================================================
m = GRX.tensor([1.0,2.0,3.0,4.0,5.0,6.0], [2,3])
assert_equal_f "get(0,0)",  1.0, m.get(0,0)
assert_equal_f "get(1,2)",  6.0, m.get(1,2)
assert_equal_f "reshape",   [2,3], m.reshape([2,3]).shape
assert_equal_f "flatten",   [1.0,2.0,3.0,4.0,5.0,6.0], m.flatten.to_a

sq = GRX.tensor([1.0,2.0,3.0,4.0], [2,2])
tr = sq.transpose
assert_equal_f "transpose shape",   [2,2], tr.shape
assert_equal_f "transpose get(0,1)",3.0,   tr.get(0,1)
assert_equal_f "transpose get(1,0)",2.0,   tr.get(1,0)
assert_equal_f "transpose to_a",    [1.0,3.0,2.0,4.0], tr.to_a

# reshape shares storage (zero-copy)
orig = GRX.tensor([1.0,2.0,3.0,4.0], [4])
view = orig.reshape([2,2])
assert_equal_f "reshape zero-copy get", 3.0, view.get(1,0)

# ================================================================
section "Activations"
# ================================================================
neg = GRX.tensor([-3.0,-1.0,0.0,1.0,3.0], [5])

relu_r = neg.relu.to_a
assert_equal_f "relu negatives → 0", [0.0,0.0,0.0,1.0,3.0], relu_r

lr = neg.leaky_relu(0.1).to_a
assert_equal_f "leaky_relu negative", -0.3, lr[0], 1e-9
assert_equal_f "leaky_relu positive",  3.0, lr[4], 1e-9

sig0 = GRX.tensor([0.0],[1]).sigmoid.to_a[0]
assert_equal_f "sigmoid(0) = 0.5", 0.5, sig0

sig_large = GRX.tensor([100.0],[1]).sigmoid.to_a[0]
assert_equal_f "sigmoid(100) ≈ 1", 1.0, sig_large, 1e-4

tanh0 = GRX.tensor([0.0],[1]).tanh.to_a[0]
assert_equal_f "tanh(0) = 0", 0.0, tanh0

sm = GRX.tensor([1.0,2.0,3.0,4.0],[4]).softmax.to_a
assert_equal_f "softmax sums to 1",  1.0, sm.sum, 1e-9
assert_equal_f "softmax all positive", true, sm.all? { |v| v > 0 }
assert_equal_f "softmax monotone",   true, sm[0] < sm[1] && sm[1] < sm[2] && sm[2] < sm[3]

# ================================================================
section "Autograd — simple ops"
# ================================================================
a_g = GRX.tensor([2.0,3.0],[2], requires_grad: true)
b_g = GRX.tensor([4.0,5.0],[2], requires_grad: true)

# addition
c = a_g + b_g; c.backward
assert_equal_f "grad add a", [1.0,1.0], a_g.grad.to_a
assert_equal_f "grad add b", [1.0,1.0], b_g.grad.to_a

# multiplication
a_g.zero_grad!; b_g.zero_grad!
d = a_g * b_g; d.backward
assert_equal_f "grad mul a (=b)", [4.0,5.0], a_g.grad.to_a
assert_equal_f "grad mul b (=a)", [2.0,3.0], b_g.grad.to_a

# subtraction
a_s = GRX.tensor([5.0,6.0],[2], requires_grad: true)
b_s = GRX.tensor([1.0,2.0],[2], requires_grad: true)
(a_s - b_s).backward
assert_equal_f "grad sub a",  [1.0,1.0],  a_s.grad.to_a
assert_equal_f "grad sub b",  [-1.0,-1.0], b_s.grad.to_a

# ================================================================
section "Autograd — chained operations"
# ================================================================
# z = (x + y) * y  →  dz/dx = y,  dz/dy = x + 2y
x_c = GRX.tensor([1.0,2.0],[2], requires_grad: true)
y_c = GRX.tensor([3.0,4.0],[2], requires_grad: true)
((x_c + y_c) * y_c).backward
assert_equal_f "chain dz/dx = y",     [3.0,4.0],  x_c.grad.to_a
assert_equal_f "chain dz/dy = x+2y",  [7.0,10.0], y_c.grad.to_a

# w = a * b + b  →  dw/da = b,  dw/db = a + 1
a_ch = GRX.tensor([2.0,3.0],[2], requires_grad: true)
b_ch = GRX.tensor([4.0,5.0],[2], requires_grad: true)
(a_ch * b_ch + b_ch).backward
assert_equal_f "chain2 dw/da = b",   [4.0,5.0], a_ch.grad.to_a
assert_equal_f "chain2 dw/db = a+1", [3.0,4.0], b_ch.grad.to_a

# ================================================================
section "Autograd — matmul"
# ================================================================
# C = A @ B,  dL/dA = dL/dC @ B^T,  dL/dB = A^T @ dL/dC
A = GRX.tensor([1.0,2.0,3.0,4.0],[2,2], requires_grad: true)
B = GRX.tensor([1.0,0.0,0.0,1.0],[2,2], requires_grad: true)  # identity
C = A.matmul(B)
C.backward
# dL/dA = I^T = I,  so grad_A = ones (grad_inicial) @ I^T = ones
assert_equal_f "matmul grad A shape", [2,2], A.grad.shape
assert_equal_f "matmul grad B shape", [2,2], B.grad.shape

# ================================================================
section "NN — Linear layer"
# ================================================================
lin = GRX::NN::Linear.new(3, 4)
assert_equal_f "Linear weight shape", [4,3], lin.weight.shape
assert_equal_f "Linear bias shape",   [4],   lin.bias.shape
assert_equal_f "Linear weight requires_grad", true, lin.weight.requires_grad
assert_equal_f "Linear bias requires_grad",   true, lin.bias.requires_grad

xin = GRX.tensor(Array.new(6){rand}, [2,3])
out = lin.call(xin)
assert_equal_f "Linear output shape", [2,4], out.shape

lin_no_bias = GRX::NN::Linear.new(3, 4, bias: false)
assert_equal_f "Linear no bias params", 1, lin_no_bias.parameters.size

# ================================================================
section "NN — Sequential & activations"
# ================================================================
net = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(4, 8),
  GRX::NN::ReLU.new,
  GRX::NN::Linear.new(8, 4),
  GRX::NN::Tanh.new,
  GRX::NN::Linear.new(4, 2),
  GRX::NN::Sigmoid.new
)
xnet = GRX.tensor(Array.new(4){rand}, [1,4])
ynet = net.call(xnet)
assert_equal_f "Sequential output shape", [1,2], ynet.shape
assert_equal_f "Sigmoid output in (0,1)", true, ynet.to_a.all? { |v| v > 0 && v < 1 }
assert_equal_f "Sequential params count", 6, net.parameters.size  # 3 weights + 3 biases

# ================================================================
section "NN — Dropout"
# ================================================================
drop = GRX::NN::Dropout.new(0.5)
drop.train!
xd = GRX.tensor(Array.new(1000, 1.0), [1000])
yd = drop.call(xd).to_a
zeros = yd.count { |v| v == 0.0 }
assert_equal_f "Dropout ~50% zeros", true, zeros.between?(350, 650)

drop.eval!
yd_eval = drop.call(xd).to_a
assert_equal_f "Dropout eval = identity", [1.0]*1000, yd_eval

# ================================================================
section "NN — Weight initialization"
# ================================================================
xav = GRX::Tensor.xavier_uniform([64, 32])
lim = Math.sqrt(6.0 / 96)
assert_equal_f "Xavier shape",    [64,32], xav.shape
assert_equal_f "Xavier in range", true, xav.to_a.all? { |v| v.abs <= lim + 1e-9 }

he = GRX::Tensor.he_normal([500, 64])
vals = he.to_a
mean = vals.sum / vals.size
std  = Math.sqrt(vals.sum { |v| (v-mean)**2 } / vals.size)
expected_std = Math.sqrt(2.0 / 64)
assert_equal_f "He shape",    [500,64], he.shape
assert_equal_f "He mean ≈ 0", true, mean.abs < 0.05
assert_equal_f "He std ≈ √(2/fan_in)", true, (std - expected_std).abs < 0.02

# ================================================================
section "Loss functions"
# ================================================================
p_perf = GRX.tensor([1.0,2.0,3.0],[3])
t_perf = GRX.tensor([1.0,2.0,3.0],[3])
p_bad  = GRX.tensor([0.0,0.0,0.0],[3])

assert_equal_f "MSE perfect",  0.0,    GRX::Loss::MSELoss.new.call(p_perf, t_perf)
assert_equal_f "MSE nonzero",  14.0/3, GRX::Loss::MSELoss.new.call(p_bad, t_perf)
assert_equal_f "MAE perfect",  0.0,    GRX::Loss::MAELoss.new.call(p_perf, t_perf)
assert_equal_f "MAE nonzero",  2.0,    GRX::Loss::MAELoss.new.call(p_bad, t_perf)

bce_pred   = GRX.tensor([0.9, 0.1], [2])
bce_target = GRX.tensor([1.0, 0.0], [2])
bce_loss   = GRX::Loss::BCELoss.new.call(bce_pred, bce_target)
assert_equal_f "BCE near-perfect < 0.15", true, bce_loss < 0.15

ce_logits = GRX.tensor([2.0, 1.0, 0.1], [3])
ce_target = GRX.tensor([1.0, 0.0, 0.0], [3])
ce_loss   = GRX::Loss::CrossEntropyLoss.new.call(ce_logits, ce_target)
assert_equal_f "CrossEntropy > 0", true, ce_loss > 0

huber = GRX::Loss::HuberLoss.new(delta: 1.0)
assert_equal_f "Huber perfect", 0.0, huber.call(p_perf, t_perf)
assert_equal_f "Huber > 0",     true, huber.call(p_bad, t_perf) > 0

# ================================================================
section "Optimizers — SGD"
# ================================================================
w_sgd = GRX.tensor([0.0], [1], requires_grad: true)
opt_sgd = GRX::Optim::SGD.new([w_sgd], lr: 0.1)
30.times do
  opt_sgd.zero_grad
  w_sgd.agregar_gradiente(GRX.tensor([2.0 * (w_sgd.to_a[0] - 3.0)], [1]))
  opt_sgd.step
end
assert_equal_f "SGD converges to 3.0", true, (w_sgd.to_a[0] - 3.0).abs < 0.05

# SGD with momentum
w_mom = GRX.tensor([0.0], [1], requires_grad: true)
opt_mom = GRX::Optim::SGD.new([w_mom], lr: 0.05, momentum: 0.9)
80.times do
  opt_mom.zero_grad
  w_mom.agregar_gradiente(GRX.tensor([2.0 * (w_mom.to_a[0] - 3.0)], [1]))
  opt_mom.step
end
assert_equal_f "SGD+momentum converges", true, (w_mom.to_a[0] - 3.0).abs < 0.05

# ================================================================
section "Optimizers — Adam"
# ================================================================
w_adam = GRX.tensor([0.0], [1], requires_grad: true)
opt_adam = GRX::Optim::Adam.new([w_adam], lr: 0.3)
100.times do
  opt_adam.zero_grad
  w_adam.agregar_gradiente(GRX.tensor([2.0 * (w_adam.to_a[0] - 3.0)], [1]))
  opt_adam.step
end
assert_equal_f "Adam converges to 3.0", true, (w_adam.to_a[0] - 3.0).abs < 0.05

# Adam with weight decay
w_wd = GRX.tensor([5.0], [1], requires_grad: true)
opt_wd = GRX::Optim::Adam.new([w_wd], lr: 0.1, weight_decay: 0.01)
100.times do
  opt_wd.zero_grad
  w_wd.agregar_gradiente(GRX.tensor([0.0], [1]))  # zero grad, only weight decay
  opt_wd.step
end
assert_equal_f "Adam weight_decay shrinks param", true, w_wd.to_a[0].abs < 5.0

# ================================================================
section "End-to-end: network training"
# ================================================================
# Regression: learn y = 2x + 1 with a Linear network
train_x = GRX.tensor((1..8).map(&:to_f), [8, 1])
train_y = GRX.tensor((1..8).map { |x| 2.0 * x + 1.0 }, [8, 1])

reg_net = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(1, 4),
  GRX::NN::Tanh.new,
  GRX::NN::Linear.new(4, 1)
)
reg_opt  = GRX::Optim::Adam.new(reg_net.parameters, lr: 0.05)
loss_fn  = GRX::Loss::MSELoss.new

losses = []
300.times do
  reg_opt.zero_grad
  pred     = reg_net.call(train_x)
  loss_val = loss_fn.call(pred, train_y)
  losses << loss_val

  grad_data = pred.to_a.zip(train_y.to_a).map { |p,t| 2.0*(p-t)/pred.numel }
  pred.agregar_gradiente(GRX.tensor(grad_data, pred.shape))
  pred.backward
  reg_opt.step
end

assert_equal_f "Network trains without error",  true, losses.size == 300
assert_equal_f "All losses are numbers",         true, losses.all? { |l| (l.is_a?(Float) || l.is_a?(GRX::Tensor)) && !l.nan? }
assert_equal_f "Loss decreased over 300 steps",  true, losses.last < losses.first
assert_equal_f "Loss converged (< 1.0)",         true, losses.last < 1.0

# ================================================================
section "Utilities — Utils"
# ================================================================
oh = GRX::Utils.one_hot([0, 2, 1], num_classes: 4)
assert_equal_f "one_hot shape", [3, 4], oh.shape
assert_equal_f "one_hot row 0", [1.0, 0.0, 0.0, 0.0], oh.to_a.slice(0, 4)
assert_equal_f "one_hot row 1", [0.0, 0.0, 1.0, 0.0], oh.to_a.slice(4, 4)
assert_equal_f "one_hot row 2", [0.0, 1.0, 0.0, 0.0], oh.to_a.slice(8, 4)

p_clip = GRX.tensor([10.0, 20.0], [2], requires_grad: true)
p_clip.agregar_gradiente(GRX.tensor([30.0, 40.0], [2]))
clipped_norm = GRX::Utils.clip_grad_norm!([p_clip], 5.0)
assert_equal_f "clip_grad_norm! total_norm", 50.0, clipped_norm.round(1)
assert_equal_f "clip_grad_norm! scaled grad", [3.0, 4.0], p_clip.grad.to_a.map(&:round)

# ================================================================
section "Error handling"
# ================================================================
begin
  GRX.tensor([1.0,2.0],[2]) + GRX.tensor([1.0,2.0,3.0],[3])
  FAIL << "ShapeError not raised"; puts "  \e[31m✗\e[0m ShapeError not raised"
rescue GRX::ShapeError
  PASS << "ShapeError on incompatible add"; puts "  \e[32m✓\e[0m ShapeError on incompatible add"
end

begin
  GRX.tensor([1.0,2.0,3.0],[3]).reshape([2,2])
  FAIL << "ArgumentError not raised"; puts "  \e[31m✗\e[0m ArgumentError not raised"
rescue ArgumentError
  PASS << "ArgumentError on bad reshape"; puts "  \e[32m✓\e[0m ArgumentError on bad reshape"
end

begin
  GRX.tensor([1.0,2.0,3.0],[3]).transpose
  FAIL << "DimensionError not raised"; puts "  \e[31m✗\e[0m DimensionError not raised"
rescue GRX::DimensionError
  PASS << "DimensionError on 1D transpose"; puts "  \e[32m✓\e[0m DimensionError on 1D transpose"
end

# ================================================================
puts "\n" + "=" * 60
total = PASS.size + FAIL.size
puts "\e[1mResult: #{PASS.size}/#{total} passed\e[0m"
if FAIL.empty?
  puts "\e[32m✓ All tests passed — ready for RubyGems 💎\e[0m"
else
  puts "\e[31m✗ #{FAIL.size} failure(s):\e[0m"
  FAIL.each { |f| puts "    - #{f}" }
end
puts "=" * 60
exit FAIL.empty? ? 0 : 1
