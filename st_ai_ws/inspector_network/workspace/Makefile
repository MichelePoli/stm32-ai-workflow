# SANITIZE_FLAGS ?= -O0 -g -fstack-protector-all -fsanitize=address -fsanitize=undefined -static-libasan -fno-omit-frame-pointer
SANITIZE_FLAGS ?=
WORKSPACE ?= .
CC ?= gcc
CFLAGS += -std=c99 -Wall -DHAS_LOG=6 -Wextra -Wdouble-promotion $(SANITIZE_FLAGS)
CFLAGS += -I$(WORKSPACE)/generated -I$(WORKSPACE)/include
LDFLAGS = -shared $(SANITIZE_FLAGS)

NAME ?= network

TARGET = libai_${NAME}
BUILD_DIR = ./$(WORKSPACE)/build
LIBS = -L$(WORKSPACE)/lib/static -lruntime -lst_cmsis_nn -lcmsis-nn -lm

# OS-specific shared library build
ifeq ($(OS),Windows_NT)
	TARGET := $(TARGET).dll
	LDFLAGS += -Wl,--out-implib,$(BUILD_DIR)/$(TARGET).a
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		TARGET := $(TARGET).dylib
	else
		TARGET := $(TARGET).so
	endif
	CFLAGS += -fPIC
endif


C_SOURCES = $(sort $(wildcard $(WORKSPACE)/generated/*.c))
REMOVE_SRCS = $(wildcard $(WORKSPACE)/generated/*_org.c)
REMOVE_SRCS += $(wildcard $(WORKSPACE)/generated/*_split.c)

SOURCES := $(filter-out $(REMOVE_SRCS), $(SOURCES))

OBJECTS = $(C_SOURCES:%.c=$(BUILD_DIR)/%.o)
DEP = $(OBJECTS:%.o=%.d)

copy_generated_files:
	mkdir -p $(WORKSPACE)/generated
	cp -f generated/network* $(WORKSPACE)/generated

.PHONY: install default all clean copy_generated_files

default: $(BUILD_DIR)/$(TARGET)
all: default

install: all
	rm -f $(WORKSPACE)/lib/$(TARGET)
	mv $(BUILD_DIR)/$(TARGET) "$(WORKSPACE)/lib"

$(BUILD_DIR)/$(TARGET): $(OBJECTS)
	mkdir -p $(@D)
	$(CC) $(LDFLAGS) $^ $(LIBS) -o $@

-include $(DEP)

$(BUILD_DIR)/%.o: %.c $(HEADERS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -c $< -o $@

clean:
	-rm -f $(BUILD_DIR)/$(TARGET) $(OBJECTS) $(DEP)
	-rm -f $(WORKSPACE)/lib/$(TARGET)
	-rm -rf $(BUILD_DIR)
