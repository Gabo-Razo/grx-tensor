# extconf.rb
# =====================================================================
# Script de configuración de la extensión nativa.
# rake-compiler lo ejecuta para generar el Makefile correcto
# según la plataforma del usuario.
#
# Uso:
#   bundle exec rake compile          → compila para la plataforma actual
#   bundle exec rake native gem       → empaqueta binarios pre-compilados
# =====================================================================

require "mkmf"

extension_name = "grx_core"

$CFLAGS << " -O3 -ffast-math"
$CFLAGS << " -fvisibility=hidden" unless RUBY_PLATFORM =~ /mingw|mswin/

if try_compile("int main(){return 0;}", "-mavx2 -mfma")
  $CFLAGS << " -mavx2 -mfma"
  puts "GRX: AVX2 + FMA habilitados"
elsif try_compile("int main(){return 0;}", "-msse4.2")
  $CFLAGS << " -msse4.2"
  puts "GRX: SSE4.2 habilitado"
else
  puts "GRX: Sin SIMD — usando implementación escalar"
end

have_library("m") unless RUBY_PLATFORM =~ /mingw|mswin/

create_makefile(extension_name)
