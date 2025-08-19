# Makefile for gaiamock - Gaia astrometry simulation package
# Based on installation instructions from the README

# Variables
PYTHON = python3
PIP = pip3
GSL_PREFIX = $(shell brew --prefix gsl 2>/dev/null || echo "/usr/local")
HEALPIX_URL = https://caltech.box.com/s/4f7q6qdh0bku881bzvzxc4cm5u0902cf
C_SOURCE = kepler_solve_astrometry.c
C_OUTPUT = kepler_solve_astrometry.so
HEALPIX_DIR = healpix_scans
HEALPIX_ZIP = healpix_scans.zip

# Detect OS for different compilation flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS
    GSL_CFLAGS = -I$(GSL_PREFIX)/include
    GSL_LDFLAGS = -L$(GSL_PREFIX)/lib
    CC_FLAGS = -shared -fPIC
else
    # Linux/Unix
    GSL_CFLAGS = $(shell pkg-config --cflags gsl 2>/dev/null || echo "-I/usr/include")
    GSL_LDFLAGS = $(shell pkg-config --libs gsl 2>/dev/null || echo "-L/usr/lib -lgsl -lgslcblas")
    CC_FLAGS = -shared -fPIC
endif

# Default target
.PHONY: all
all: install

# Complete installation process
.PHONY: install
install: check-python install-deps check-gsl download-healpix compile-c
	@echo "Installation complete! You can now use gaiamock."
	@echo "Check out demo.ipynb for basic functionality examples."

# Check Python installation
.PHONY: check-python
check-python:
	@echo "Checking Python installation..."
	@$(PYTHON) --version || (echo "Python 3 not found. Please install Python 3."; exit 1)
	@$(PIP) --version || (echo "pip not found. Please install pip."; exit 1)

# Install required Python packages
.PHONY: install-deps
install-deps:
	@echo "Installing required Python packages..."
	$(PIP) install numpy matplotlib healpy joblib

# Install optional mwdust package for dust simulation
.PHONY: install-mwdust
install-mwdust:
	@echo "Installing optional mwdust package..."
	$(PIP) install mwdust

# Check GSL installation
.PHONY: check-gsl
check-gsl:
	@echo "Checking GSL installation..."
ifeq ($(UNAME_S),Darwin)
	@if ! brew list gsl >/dev/null 2>&1; then \
		echo "GSL not found. Installing via Homebrew..."; \
		brew install gsl; \
	else \
		echo "GSL found at $(GSL_PREFIX)"; \
	fi
else
	@if ! pkg-config --exists gsl; then \
		echo "GSL not found. Please install GSL development libraries."; \
		echo "On Ubuntu/Debian: sudo apt-get install libgsl-dev"; \
		echo "On CentOS/RHEL: sudo yum install gsl-devel"; \
		exit 1; \
	else \
		echo "GSL found"; \
	fi
endif

# Download and extract healpix data
.PHONY: download-healpix
download-healpix: $(HEALPIX_DIR)

$(HEALPIX_DIR):
	@echo "Downloading healpix_scans.zip..."
	@echo "Note: This is a large file (984 MB after extraction)"
	@if [ ! -f $(HEALPIX_ZIP) ]; then \
		echo "Please download $(HEALPIX_ZIP) manually from:"; \
		echo "$(HEALPIX_URL)"; \
		echo "and place it in the current directory, then run 'make extract-healpix'"; \
		exit 1; \
	fi
	$(MAKE) extract-healpix

# Extract healpix data (assumes zip file is already downloaded)
.PHONY: extract-healpix
extract-healpix:
	@if [ ! -f $(HEALPIX_ZIP) ]; then \
		echo "$(HEALPIX_ZIP) not found. Please download it first."; \
		exit 1; \
	fi
	@echo "Extracting $(HEALPIX_ZIP)..."
	unzip -q $(HEALPIX_ZIP)
	@echo "Verifying extraction..."
	@if [ -d $(HEALPIX_DIR) ] && [ $$(ls $(HEALPIX_DIR)/*.fits 2>/dev/null | wc -l) -eq 49152 ]; then \
		echo "Successfully extracted 49152 FITS files to $(HEALPIX_DIR)/"; \
	else \
		echo "Error: Expected 49152 FITS files in $(HEALPIX_DIR)/"; \
		exit 1; \
	fi

# Compile the C extension
.PHONY: compile-c
compile-c: $(C_OUTPUT)

$(C_OUTPUT): $(C_SOURCE)
	@echo "Compiling $(C_SOURCE)..."
	gcc $(CC_FLAGS) -o $@ $< $(GSL_CFLAGS) $(GSL_LDFLAGS) -lgsl -lgslcblas -lm
	@echo "Successfully compiled $(C_OUTPUT)"

# Clean compiled files
.PHONY: clean
clean:
	@echo "Cleaning compiled files..."
	rm -f $(C_OUTPUT)

# Clean all generated/downloaded files
.PHONY: clean-all
clean-all: clean
	@echo "Cleaning all generated files..."
	rm -rf $(HEALPIX_DIR)
	rm -f $(HEALPIX_ZIP)

# Development setup (includes optional dependencies)
.PHONY: dev-install
dev-install: install install-mwdust
	@echo "Development installation complete!"

# Test the installation
.PHONY: test
test:
	@echo "Testing installation..."
	@if [ ! -f $(C_OUTPUT) ]; then \
		echo "Error: $(C_OUTPUT) not found. Run 'make compile-c'"; \
		exit 1; \
	fi
	@if [ ! -d $(HEALPIX_DIR) ]; then \
		echo "Error: $(HEALPIX_DIR) directory not found. Run 'make download-healpix'"; \
		exit 1; \
	fi
	@$(PYTHON) -c "import numpy, matplotlib, healpy, joblib; print('All required packages imported successfully')"
	@echo "Installation test passed!"

# Help target
.PHONY: help
help:
	@echo "Gaiamock Makefile - Available targets:"
	@echo ""
	@echo "  install         - Complete installation (default)"
	@echo "  install-deps    - Install Python dependencies only"
	@echo "  install-mwdust  - Install optional mwdust package"
	@echo "  check-gsl       - Check/install GSL library"
	@echo "  download-healpix- Download and extract healpix data"
	@echo "  extract-healpix - Extract healpix data (if already downloaded)"
	@echo "  compile-c       - Compile C extension"
	@echo "  dev-install     - Development installation (includes optional deps)"
	@echo "  test            - Test the installation"
	@echo "  clean           - Remove compiled files"
	@echo "  clean-all       - Remove all generated/downloaded files"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Notes:"
	@echo "  - The healpix_scans.zip file must be downloaded manually from:"
	@echo "    $(HEALPIX_URL)"
	@echo "  - On macOS, GSL will be installed via Homebrew if not present"
	@echo "  - On Linux, you may need to install GSL dev packages manually"
