# =============================================================
# Makefile — Linux / macOS
#
# Compila grx_core.c DIRECTAMENTE en lib/grx/ (sin archivo intermedio).
# No hay .so en ext/unix/ — el único binario vive en lib/grx/.
#
# Uso:
#   make          → compila (detecta OS y SIMD automáticamente)
#   make clean    → elimina el binario de lib/grx/
#   make bench    → compila y corre el benchmark
# =============================================================

CC      = gcc
CFLAGS  = -O3 -march=native -ffast-math -funroll-loops \
          -fPIC -fvisibility=hidden \
          -Wall -Wextra -std=c11
LDFLAGS = -lm
SRC_DIR = ../grx
SRC     = $(SRC_DIR)/grx_core.c
HEADER  = $(SRC_DIR)/grx_core.h

# Destino final — directamente en lib/grx/
OUT_DIR = ../../lib/grx

UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
    LIB = libgrx_core.dylib
    SHARED = -dynamiclib
else
    LIB = libgrx_core.so
    SHARED = -shared
endif

TARGET = $(OUT_DIR)/$(LIB)

# Detección de SIMD
AVX2_TEST := $(shell echo 'int main(){}' | $(CC) -mavx2 -mfma -x c - -o /dev/null 2>&1)
ifeq ($(AVX2_TEST),)
    CFLAGS += -mavx2 -mfma
    $(info [GRX] AVX2 + FMA habilitados — máxima velocidad SIMD)
else
    SSE2_TEST := $(shell echo 'int main(){}' | $(CC) -msse2 -x c - -o /dev/null 2>&1)
    ifeq ($(SSE2_TEST),)
        CFLAGS += -msse2
        $(info [GRX] SSE2 habilitado)
    else
        $(info [GRX] Sin SIMD — modo escalar)
    endif
endif

.PHONY: all clean bench

all: $(TARGET)

$(TARGET): $(SRC) $(HEADER)
	@mkdir -p $(OUT_DIR)
	$(CC) $(CFLAGS) $(SHARED) $(SRC) -o $(TARGET) $(LDFLAGS)
	@echo "[GRX] Compilado → $(TARGET)"

bench: all
	@echo "[GRX] Corriendo benchmark..."
	ruby -I../../lib ../../test/benchmark.rb

clean:
	rm -f $(TARGET)
	@echo "[GRX] Limpiado: $(TARGET)"
