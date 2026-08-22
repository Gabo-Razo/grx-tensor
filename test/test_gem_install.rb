# Test de integración usando la gema instalada (no el directorio local)
# Corre con: ruby test/test_gem_install.rb  (SIN -Ilib)

require "grx"

passed = 0
failed = 0

def check(desc, expected, actual, delta = 1e-6)
  ok = if expected.is_a?(Array)
    expected.zip(actual).all? { |e, a| (e - a).abs < delta }
  elsif expected == true || expected == false
    expected == actual
  else
    (expected - actual).abs < delta
  end
  if ok
    puts "  ✓ #{desc}"
    true
  else
    puts "  ✗ #{desc} — expected #{expected.inspect}, got #{actual.inspect}"
    false
  end
end

puts "=" * 55
puts "GRX-Tensor #{GRX::VERSION} — gem install integration test"
puts "Mode: #{GRX::CAPI::LOADED ? 'C + SIMD ⚡' : 'Ruby fallback'}"
puts "=" * 55

# ------------------------------------------------------------------
puts "\n[ Arithmetic ]"
# ------------------------------------------------------------------
a = GRX.tensor([1.0, 2.0, 3.0], [3])
b = GRX.tensor([4.0, 5.0, 6.0], [3])

passed += check("add",    [5.0, 7.0, 9.0],    (a + b).to_a) ? 1 : (failed += 1; 0)
passed += check("sub",    [-3.0, -3.0, -3.0], (a - b).to_a) ? 1 : (failed += 1; 0)
passed += check("mul",    [4.0, 10.0, 18.0],  (a * b).to_a) ? 1 : (failed += 1; 0)
passed += check("div",    [0.25, 0.4, 0.5],   (a / b).to_a) ? 1 : (failed += 1; 0)
passed += check("dot",    32.0,               a.dot(b))      ? 1 : (failed += 1; 0)
passed += check("negate", [-1.0,-2.0,-3.0],   a.negate.to_a) ? 1 : (failed += 1; 0)

# ------------------------------------------------------------------
puts "\n[ Math ops ]"
# ------------------------------------------------------------------
passed += check("sqrt",   [2.0,3.0,4.0],      GRX.tensor([4.0,9.0,16.0],[3]).sqrt.to_a)   ? 1 : (failed+=1;0)
passed += check("square", [1.0,4.0,9.0],      a.square.to_a)                               ? 1 : (failed+=1;0)
passed += check("abs",    [1.0,2.0,3.0],      a.negate.abs.to_a)                           ? 1 : (failed+=1;0)
passed += check("clip",   [1.5,2.0,2.5],      a.clip(1.5,2.5).to_a)                        ? 1 : (failed+=1;0)
passed += check("pow(3)", [1.0,8.0,27.0],     a.pow(3).to_a)                               ? 1 : (failed+=1;0)
passed += check("exp/log roundtrip", [1.0,2.0,3.0], a.log.exp.to_a)                        ? 1 : (failed+=1;0)

# ------------------------------------------------------------------
puts "\n[ Reductions ]"
# ------------------------------------------------------------------
r = GRX.tensor([1.0,2.0,3.0,4.0],[4])
passed += check("sum",  10.0, r.sum)  ? 1 : (failed+=1;0)
passed += check("mean", 2.5,  r.mean) ? 1 : (failed+=1;0)
passed += check("max",  4.0,  r.max)  ? 1 : (failed+=1;0)
passed += check("min",  1.0,  r.min)  ? 1 : (failed+=1;0)

# ------------------------------------------------------------------
puts "\n[ Activations ]"
# ------------------------------------------------------------------
v = GRX.tensor([-2.0, 0.0, 2.0], [3])
passed += check("relu",       [0.0,0.0,2.0], v.relu.to_a)                                  ? 1 : (failed+=1;0)
passed += check("leaky_relu", [-0.02,0.0,2.0], v.leaky_relu(0.01).to_a)                    ? 1 : (failed+=1;0)
passed += check("sigmoid(0)", 0.5, GRX.tensor([0.0],[1]).sigmoid.to_a[0])                  ? 1 : (failed+=1;0)
passed += check("tanh(0)",    0.0, GRX.tensor([0.0],[1]).tanh.to_a[0])                     ? 1 : (failed+=1;0)
passed += check("softmax sum",1.0, GRX.tensor([1.0,2.0,3.0],[3]).softmax.to_a.sum)         ? 1 : (failed+=1;0)

# ------------------------------------------------------------------
puts "\n[ Geometry ]"
# ------------------------------------------------------------------
m = GRX.tensor([1.0,2.0,3.0,4.0],[2,2])
passed += check("get(1,0)",  3.0,                m.get(1,0))                                ? 1 : (failed+=1;0)
passed += check("transpose", [1.0,3.0,2.0,4.0], m.transpose.to_a)                          ? 1 : (failed+=1;0)
passed += check("reshape",   [1.0,2.0,3.0,4.0], m.reshape([4]).to_a)                       ? 1 : (failed+=1;0)
passed += check("matmul",    [19.0,22.0,43.0,50.0],
  GRX.tensor([1.0,2.0,3.0,4.0],[2,2]).matmul(GRX.tensor([5.0,6.0,7.0,8.0],[2,2])).to_a)   ? 1 : (failed+=1;0)

# ------------------------------------------------------------------
puts "\n[ Autograd ]"
# ------------------------------------------------------------------
a2 = GRX.tensor([2.0,3.0],[2], requires_grad: true)
b2 = GRX.tensor([4.0,5.0],[2], requires_grad: true)
(a2 + b2).backward
passed += check("grad +  a", [1.0,1.0], a2.grad.to_a) ? 1 : (failed+=1;0)
passed += check("grad +  b", [1.0,1.0], b2.grad.to_a) ? 1 : (failed+=1;0)

a3 = GRX.tensor([2.0,3.0],[2], requires_grad: true)
b3 = GRX.tensor([4.0,5.0],[2], requires_grad: true)
(a3 * b3).backward
passed += check("grad *  a", [4.0,5.0], a3.grad.to_a) ? 1 : (failed+=1;0)
passed += check("grad *  b", [2.0,3.0], b3.grad.to_a) ? 1 : (failed+=1;0)

x = GRX.tensor([1.0,2.0],[2], requires_grad: true)
y = GRX.tensor([3.0,4.0],[2], requires_grad: true)
((x + y) * y).backward
passed += check("grad chain x", [3.0,4.0],  x.grad.to_a) ? 1 : (failed+=1;0)
passed += check("grad chain y", [7.0,10.0], y.grad.to_a) ? 1 : (failed+=1;0)

# ------------------------------------------------------------------
puts "\n[ NN layers ]"
# ------------------------------------------------------------------
net = GRX::NN::Sequential.new(
  GRX::NN::Linear.new(4, 8),
  GRX::NN::ReLU.new,
  GRX::NN::Linear.new(8, 2)
)
out = net.call(GRX.tensor(Array.new(4){rand},[1,4]))
passed += check("Sequential output shape", [1,2], out.shape) ? 1 : (failed+=1;0)
passed += check("parameters count", 4, net.parameters.size)  ? 1 : (failed+=1;0)

# ------------------------------------------------------------------
puts "\n[ Loss functions ]"
# ------------------------------------------------------------------
p1 = GRX.tensor([1.0,2.0,3.0],[3]); t1 = GRX.tensor([1.0,2.0,3.0],[3])
passed += check("MSE perfect", 0.0,    GRX::Loss::MSELoss.new.call(p1,t1)) ? 1 : (failed+=1;0)
passed += check("MAE perfect", 0.0,    GRX::Loss::MAELoss.new.call(p1,t1)) ? 1 : (failed+=1;0)
passed += check("MSE nonzero", 14.0/3, GRX::Loss::MSELoss.new.call(GRX.tensor([0.0,0.0,0.0],[3]),t1)) ? 1 : (failed+=1;0)

# ------------------------------------------------------------------
puts "\n[ Optimizers ]"
# ------------------------------------------------------------------
w = GRX.tensor([0.0],[1], requires_grad: true)
opt = GRX::Optim::SGD.new([w], lr: 0.1)
20.times { opt.zero_grad; w.agregar_gradiente(GRX.tensor([2.0*(w.to_a[0]-2.0)],[1])); opt.step }
passed += check("SGD converges to 2.0", true, (w.to_a[0]-2.0).abs < 0.1) ? 1 : (failed+=1;0)

w2 = GRX.tensor([0.0],[1], requires_grad: true)
opt2 = GRX::Optim::Adam.new([w2], lr: 0.3)
50.times { opt2.zero_grad; w2.agregar_gradiente(GRX.tensor([2.0*(w2.to_a[0]-2.0)],[1])); opt2.step }
passed += check("Adam converges to 2.0", true, (w2.to_a[0]-2.0).abs < 0.1) ? 1 : (failed+=1;0)

# ------------------------------------------------------------------
puts "\n[ Weight initialization ]"
# ------------------------------------------------------------------
xav = GRX::Tensor.xavier_uniform([100,100])
lim = Math.sqrt(6.0/200)
passed += check("Xavier in range", true, xav.to_a.all?{|v| v.abs <= lim+1e-9}) ? 1 : (failed+=1;0)

he = GRX::Tensor.he_normal([500,100])
mean = he.to_a.sum / he.numel
passed += check("He mean ~0", true, mean.abs < 0.05) ? 1 : (failed+=1;0)

# ------------------------------------------------------------------
puts "\n" + "=" * 55
puts "Result: #{passed}/#{passed+failed} passed"
if failed == 0
  puts "✓ All tests passed — ready for RubyGems \u{1F48E}"
else
  puts "✗ #{failed} failure(s) — fix before publishing"
end
puts "=" * 55
