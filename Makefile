################################################################################
# Makefile
#
#   * General
#   * Output Dirs
#   * Environment
#   * Articles
#   * Theme
#   * Site
#   * Tests
#   * Linters
#   * Phonies
#
################################################################################

# Verify environment.sh
ifneq ($(PROJECT_NAME),llama-kit)
$(error Environment not configured. Run `source environment.sh`)
endif


################################################################################
# Settings
################################################################################


#-------------------------------------------------------------------------------
# General
#-------------------------------------------------------------------------------

# Bash
export SHELL := /bin/bash
.SHELLFLAGS := -e -u -o pipefail -c

# Colors - Supports colorized messages
COLOR_H1=\033[38;5;12m
COLOR_OK=\033[38;5;02m
COLOR_COMMENT=\033[38;5;08m
COLOR_RESET=\033[0m

# EXCLUDE_SRC - Source patterns to ignore

EXCLUDE_SRC := __pycache__ \
			   .egg-info \
			   .ipynb_checkpoints \
			   .venv
EXCLUDE_SRC := $(subst $(eval ) ,|,$(EXCLUDE_SRC))

# Commands
RM := rm -rf


#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

OUTPUT_DIRS :=

BUILD_DIR := $(PROJECT_ROOT)/.build
OUTPUT_DIRS := $(OUTPUT_DIRS) $(BUILD_DIR)


#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

VENV_ROOT := .venv
VENV_SRC := pyproject.toml uv.lock
VENV := $(VENV_ROOT)/bin/activate


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

PYTEST_OPTS ?= 


#-------------------------------------------------------------------------------
# Linters
#-------------------------------------------------------------------------------

RUFF_CHECK_OPTS ?= --preview
RUFF_FORMAT_OPTS ?= --preview


#-------------------------------------------------------------------------------
# Phonies
#-------------------------------------------------------------------------------

PHONIES :=


################################################################################
# Targets
################################################################################

all: venv
	@echo
	@echo -e "$(COLOR_H1)# $(PROJECT_NAME)$(COLOR_RESET)"
	@echo
	@echo -e "$(COLOR_COMMENT)# Activate VENV$(COLOR_RESET)"
	@echo -e "source $(VENV)"
	@echo
	@echo -e "$(COLOR_COMMENT)# Deactivate VENV$(COLOR_RESET)"
	@echo -e "deactivate"
	@echo


#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $@


#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

$(VENV): $(VENV_SRC)
	uv sync
	
	touch $@

venv: $(VENV)
PHONIES := $(PHONIES) venv


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

tests: $(VENV)
	@echo
	@echo -e "$(COLOR_H1)# Tests$(COLOR_RESET)"
	@echo

	source $(VENV) && pytest $(PYTEST_OPTS) tests

coverage: $(VENV)
	@echo
	@echo -e "$(COLOR_H1)# Coverage$(COLOR_RESET)"
	@echo
	mkdir -p $$(dirname $(BUILD_DIR)/coverage)
	source $(VENV) && pytest $(PYTEST_OPTS) --cov=xformers --cov-report=html:$(BUILD_DIR)/coverage tests

PHONIES := $(PHONIES) tests coverage


#-------------------------------------------------------------------------------
# Linters
#-------------------------------------------------------------------------------

lint: venv
	uvx ruff check $(RUFF_CHECK_OPTS)
	uvx ruff format --check $(RUFF_FORMAT_OPTS)

lint-fix: venv
	uvx ruff format $(RUFF_FORMAT_OPTS)
	uvx ruff check --fix $(RUFF_CHECK_OPTS)
	make lint

PHONIES := $(PHONIES) lint-fix lint


#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-venv:
	$(RM) "$(VENV_ROOT)"

clean-build:
	$(RM) "$(BUILD_DIR)"

clean: clean-cache clean-venv clean-build 
PHONIES := $(PHONIES) clean-cache clean-venv clean-build clean

.PHONY: $(PHONIES)
