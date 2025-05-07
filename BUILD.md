# Building AOCL-DLP (Deep Learning Primitives)

This document provides instructions for building the AOCL-DLP library from source code.

## 📋 System Requirements

Before building AOCL-DLP, ensure your system meets the following requirements:

### Software
- CMake (≥ 3.26)
- C/C++ compiler with C11/C++17 support (e.g., GCC 11+, Clang 14+)
- OpenMP and/or pthread libraries (for multi-threading)
- ninja-build (optional, for Ninja generator support)

**Note: GCC 11 introduced AVX512_BF16 support, which is required for bfloat16 GEMM.**

### Hardware
- x86 CPU with AVX2/FMA3 support
- AVX512 support for enhanced performance
- AVX512_VNNI support for int8 GEMM
- AVX512_BF16 support for bfloat16 GEMM

## Build Configuration Options

AOCL-DLP uses CMake for its build system with several configurable options:

| Option             | Default | Description                                   |
|--------------------|---------|-----------------------------------------------|
| BUILD_EXAMPLES     | OFF     | Build example programs                         |
| BUILD_BENCHMARKS   | OFF     | Build benchmark programs                       |
| DLP_THREADING_MODEL| "none"  | Threading model ("none", "openmp", "pthread") |
| DLP_OPENMP_ROOT    | ""      | Custom path to OpenMP installation            |
| DLP_USE_LLVM_OPENMP| OFF     | Force using LLVM OpenMP implementation        |

## Quick Start Build

### Linux

1. Clone and enter project:

   ```bash
   git clone <repository-url> && cd aocl-dlp
   ```

2. Create an out-of-source build directory:

   ```bash
   mkdir -p build && cd build
   ```

3. Configure (choose generator):
   ```bash
   # Default (GNU Make)
   cmake ..

   # Ninja (fast incremental builds)
   cmake -G Ninja ..
   ```

4. Build:
   ```bash
   # Make
   make -j$(nproc)

   # Ninja
   ninja
   ```

5. For installation instructions, see `INSTALL.md`.

---

## Advanced Build Configuration

### Enabling Additional Components

To enable benchmarks, use the following CMake options:

```bash
cmake -DBUILD_BENCHMARKS=ON ..
```

### Threading Model Configuration

AOCL-DLP supports multiple threading models:

```bash
# No threading (default)
cmake -DDLP_THREADING_MODEL=none ..

# Enable OpenMP threading
cmake -DDLP_THREADING_MODEL=openmp ..

# Enable Pthread threading
cmake -DDLP_THREADING_MODEL=pthread ..
```

For custom OpenMP installation:

```bash
cmake -DDLP_THREADING_MODEL=openmp -DDLP_OPENMP_ROOT=/path/to/openmp ..
```

### Specifying Build Type

You can specify different build types:

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (default)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Release with debug info
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

## 🏁 Benchmarking

Enable and run tests and benchmarks in one place:

```bash
# Configure with tests and benchmarks
cmake -DBUILD_BENCHMARKS=ON ..

# Build (make or ninja)
make -j$(nproc)
# or
ninja

# Run benchmarks
cd ../
./build/bench/classic/bench_lpgemm  -m p -n 50 -i bench/classic/bench_input.txt
```

Input files for benchmarks are available in the `bench/classic` directory:
- bench_input.txt
- bench_batch_input.txt
- bench_sym_quant_input.txt
- bench_unpack_input.txt
- bench_utils_input.txt
- bench_eltwise_ops_input.txt

## Developer Tips

- **Out-of-tree builds**: Always build in a separate `build/` directory to keep sources clean.
- **Custom install prefix**:
  ```bash
  cmake -DCMAKE_INSTALL_PREFIX=/opt/aocl-dlp ..
  ```
- **Verbose output**:
  ```bash
  # Make
  make VERBOSE=1

  # Ninja
  ninja -v
  ```
- **Clean cache**:
  ```bash
  rm -rf build/* && cd build && cmake ..
  ```
- **Parallel builds**: Leverage all cores with `-j$(nproc)` (Make) or default Ninja parallelism.

## CMake Build System Overview

AOCL-DLP uses a modern CMake build system structured as follows:

- Main `CMakeLists.txt`: Orchestrates the overall build process
- `cmake/dlp_variables.cmake`: Sets project variables, languages and standards
- `cmake/dlp_options.cmake`: Defines build options and threading models
- `cmake/dlp_dependencies.cmake`: Manages OpenMP and Pthread dependencies
- `cmake/dlp_compiler_flags_linux.cmake`: Sets compiler flags for Linux
- `cmake/dlp_compiler_flags_windows.cmake`: Sets compiler flags for Windows
- `cmake/dlp_install.cmake`: Manages installation rules
- `cmake/dlp_extensions.cmake`: Defines file extensions

## Troubleshooting

### Threading Model Issues

If you encounter issues with the selected threading model:

1. Make sure the required libraries are installed on your system:
   - For OpenMP: OpenMP development libraries
   - For Pthread: POSIX threads library

2. For OpenMP-specific issues:

```bash
cmake -DDLP_THREADING_MODEL=openmp -DDLP_OPENMP_ROOT=/path/to/openmp ..
```

### Compiler Requirements

Make sure your compiler supports:
- C11 standard for C code
- C++17 standard for C++ code

### Build Performance

To speed up the build process, use parallel compilation:

```bash
make -j$(nproc)  # Linux
```

## Known Issues

- Warnings may appear during compilation (-Werror is currently disabled)
- Some platforms may require specific environment setup for threading model detection
