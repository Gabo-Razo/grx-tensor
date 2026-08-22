# frozen_string_literal: true

require_relative "../lib/grx"
require "benchmark"

puts "=" * 60
puts "GRX Benchmark — #{RUBY_DESCRIPTION}"
puts "Modo: #{GRX::CAPI::LOADED ? 'C + SIMD' : 'Ruby puro (fallback)'}"
puts "=" * 60

SIZES = [1_000, 100_000, 1_000_000]

SIZES.each do |n|
  data_a = Array.new(n) { rand }
  data_b = Array.new(n) { rand }

  a = GRX.tensor(data_a, [n])
  b = GRX.tensor(data_b, [n])

  iters = n <= 1_000 ? 10_000 : (n <= 100_000 ? 500 : 50)

  puts "\n--- n = #{n.to_s.rjust(9)} elementos (#{iters} iteraciones) ---"

  Benchmark.bm(12) do |x|
    x.report("add:     ") { iters.times { a + b } }
    x.report("sub:     ") { iters.times { a - b } }
    x.report("mul:     ") { iters.times { a * b } }
    x.report("dot:     ") { iters.times { a.dot(b) } }
    x.report("scale:   ") { iters.times { a.scale(2.5) } }
    x.report("relu:    ") { iters.times { a.relu } }
  end
end

# Benchmark matmul
puts "\n--- matmul ---"
[32, 128, 256].each do |sz|
  data = Array.new(sz * sz) { rand }
  a = GRX.tensor(data, [sz, sz])
  b = GRX.tensor(data, [sz, sz])
  iters = sz <= 32 ? 500 : (sz <= 128 ? 20 : 5)
  Benchmark.bm(20) do |x|
    x.report("matmul #{sz}x#{sz} (#{iters}x):") { iters.times { a.matmul(b) } }
  end
end

puts "\n" + "=" * 60
puts "Benchmark completado."
