# frozen_string_literal: true

require "bundler/gem_tasks"
require "rake/extensiontask"
require "rake/testtask"

gemspec = Gem::Specification.load("grx-tensor.gemspec")

Rake::ExtensionTask.new("grx_core", gemspec) do |ext|
  ext.lib_dir = "lib/grx"
  ext.ext_dir = "ext/grx"

  # Cross-compilación para Windows desde Linux (requiere mingw-w64)
  # Uso: bundle exec rake native:x86_64-mingw32 gem
  ext.cross_compile  = true
  ext.cross_platform = %w[x86_64-mingw32 x64-mingw-ucrt]
end

Rake::TestTask.new(:test) do |t|
  t.libs    << "test"
  t.libs    << "lib"
  t.pattern = "test/**/*_test.rb"
  t.verbose = true
end

desc "Compila con el Makefile Unix manual (sin rake-compiler)"
task :build_unix do
  sh "make -C ext/unix install"
end

desc "Compila con el Makefile Windows (requiere MinGW en PATH)"
task :build_windows do
  sh "make -C ext/windows -f Makefile.mingw install"
end

task default: %i[compile test]
