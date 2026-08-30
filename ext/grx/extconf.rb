# frozen_string_literal: true

# =====================================================================
# extconf.rb — Configuracion de compilacion estandar de RubyGems
# =====================================================================

require "mkmf"

extension_name = "grx_core"

$CFLAGS << " -O3 -ffast-math"
$CFLAGS << " -fPIC" unless RUBY_PLATFORM =~ /mingw|mswin/
$CFLAGS << " -fvisibility=hidden" unless RUBY_PLATFORM =~ /mingw|mswin/

have_library("m") unless RUBY_PLATFORM =~ /mingw|mswin/

create_makefile(extension_name)
