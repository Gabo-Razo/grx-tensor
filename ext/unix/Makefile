# =============================================================
# Makefile — Linux / macOS
# =============================================================

CC      ?= gcc
CFLAGS  = -O3 -ffast-math -funroll-loops \
          -fPIC -fvisibility=hidden \
          -Wall -Wextra -std=c11
LDFLAGS = -lm
SRC_DIR = ../grx
SRC     = $(SRC_DIR)/grx_core.c
HEADER  = $(SRC_DIR)/grx_core.h

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

.PHONY: all clean bench

all: $(TARGET)

$(TARGET): $(SRC) $(HEADER)
	@mkdir -p $(OUT_DIR)
	$(CC) $(CFLAGS) $(SHARED) $(SRC) -o $(TARGET) $(LDFLAGS)
	@echo "[GRX] Compilado exitosamente → $(TARGET)"

bench: all
	@echo "[GRX] Corriendo benchmark..."
	ruby -I../../lib ../../test/benchmark.rb

clean:
	rm -f $(TARGET)
	@echo "[GRX] Limpiado: $(TARGET)"
