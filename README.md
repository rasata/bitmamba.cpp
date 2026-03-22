# BitMamba.cpp — ARM/Apple Silicon Port

> **Fork of [Zhayr1/bitmamba.cpp](https://github.com/Zhayr1/bitmamba.cpp)** with ARM NEON support for Apple Silicon and ARM-based processors.

This fork ports the BitMamba-2 inference engine from x86-only (AVX2) to a **cross-platform implementation** supporting both x86 (AVX2) and ARM (NEON), demonstrating the **O(1) memory property of State Space Models** on non-GPU architectures.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/Paper-Zenodo-00649C.svg)](https://doi.org/10.5281/zenodo.18394665)
[![Hugging Face 1B](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-1B%20Model-FFD21E)](https://huggingface.co/Zhayr1/BitMamba-2-1B)
[![Hugging Face 255M](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-255M%20Model-FFD21E)](https://huggingface.co/Zhayr1/BitMamba-2-0.25B)

## What's new in this fork

- **ARM NEON support**: Full port of AVX2 SIMD kernels to ARM NEON intrinsics
- **Apple Silicon compilation**: Builds and runs natively on Apple M1/M2/M3/M4
- **Cross-platform dispatch**: Compile-time `#ifdef` selects x86 or ARM codepath
- **OpenMP on macOS**: CMake configured for AppleClang + Homebrew libomp
- **Benchmark data**: First published ARM benchmark results for BitMamba-2

## Benchmark Results — Apple M1 (ARM NEON)

| Model | Weights | Speed | Latency/token | Prefill (8 tok) | Stable over length? |
|-------|---------|-------|---------------|-----------------|---------------------|
| BitMamba-2 **255M** | 246 MB | **82.5 tok/s** | 12.1 ms | 69.6 ms | Yes |
| BitMamba-2 **1B** | 614 MB | **27.9 tok/s** | 35.9 ms | 242 ms | Yes |

**Key finding**: Speed is **perfectly constant** regardless of sequence length (50, 200, or more tokens). This experimentally validates the **O(1) memory** property of SSM architectures — unlike Transformers whose memory grows with sequence length.

### Comparison with references

| Model | Hardware | tok/s |
|-------|----------|-------|
| **BitMamba-2 255M** | **Apple M1 (NEON)** | **82.5** |
| **BitMamba-2 1B** | **Apple M1 (NEON)** | **27.9** |
| BitMamba-2 1B | x86 consumer (AVX2) | ~50 (original repo) |
| llama.cpp Llama-7B Q4 | Apple M1 | ~15 |
| Claude 3.5 Haiku | Cloud GPU | 61 |
| GPT-4o Mini | Cloud GPU | 59 |

## Why this matters

State Space Models (Mamba, RWKV) represent a **mathematical reformulation** of neural network computation that is structurally advantageous for non-GPU architectures:

1. **Sequential recurrence** `h_t = A h_{t-1} + B x_t` is CPU-native (no parallelism needed)
2. **O(1) memory** — state size is constant, independent of sequence length
3. **1.58-bit quantization** — ternary weights {-1, 0, +1} replace multiplications with additions
4. **Cache-friendly** — small state vectors fit in L1/L2 cache

This fork is part of the [Aquantic Research](https://github.com/rasata/aquantic-research-gpu-to-cpu-transposition) programme on mathematical reformulations of neural networks for CPU/ARM architectures.

## Requirements and compatibility

| Architecture | SIMD | Status |
|-------------|------|--------|
| x86_64 (Intel/AMD) | AVX2 | Supported (original) |
| **ARM64 (Apple Silicon)** | **NEON** | **Supported (this fork)** |
| ARM64 (Raspberry Pi 4/5) | NEON | Should work (untested) |
| ARM64 (AWS Graviton) | NEON | Should work (untested) |

## Quick Start

### 1. Build

```bash
# macOS (Apple Silicon) — requires Homebrew libomp
brew install libomp
cmake -B build
cmake --build build

# Linux x86
cmake -B build
cmake --build build
```

### 2. Download weights

```bash
# 255M model (246 MB)
wget https://huggingface.co/Zhayr1/BitMamba-2-0.25B/resolve/main/bitmamba_cpp/bitmamba_255m.bin

# 1B model (614 MB)
wget https://huggingface.co/Zhayr1/BitMamba-2-1B/resolve/main/bitmamba_cpp/bitmamba_1b.bin
```

### 3. Run inference

```bash
cd build
cp ../tokenizer.bin .

# 255M model
./bitmamba ../bitmamba_255m.bin "The future of AI is" tokenizer 0.7 1.1 0.05 0.9 40 200

# 1B model
./bitmamba ../bitmamba_1b.bin "The future of AI is" tokenizer 0.7 1.1 0.05 0.9 40 200
```

## Technical Details — ARM NEON Port

### Changes from upstream

**`src/kernels.cpp`** — Core SIMD kernels:

```cpp
#if defined(__aarch64__) || defined(__ARM_NEON)
    #include <arm_neon.h>
    // float32x4_t for rms_norm (4-wide vs 8-wide AVX2)
    // int8x16_t for ternary matmul (16-wide vs 32-wide AVX2)
    // Ternary mul via: vceqq_s8 + vandq_s8 + vnegq_s8
#elif defined(__x86_64__)
    #include <immintrin.h>
    // Original AVX2 implementation
#endif
```

**`CMakeLists.txt`** — Platform-adaptive build:
- Apple: `-mcpu=native -Xclang -fopenmp` + Homebrew libomp
- Linux: `-march=native -fopenmp` (original)

**`examples/main.cpp`** — Removed unused `#include <immintrin.h>`

### NEON kernel mapping

| Operation | AVX2 (x86) | NEON (ARM) |
|-----------|-----------|------------|
| Float32 FMA | `_mm256_fmadd_ps` (8-wide) | `vfmaq_f32` (4-wide) |
| Float32 mul | `_mm256_mul_ps` (8-wide) | `vmulq_f32` (4-wide) |
| Int8 load | `_mm256_loadu_si256` (32B) | `vld1q_s8` (16B) |
| Horizontal sum | manual 8-elem reduce | `vaddvq_f32` / `vaddvq_s32` |
| Ternary matmul | `_mm256_sign_epi8` | mask + negate + add |
| Widen int8->int16 | `_mm256_cvtepi8_epi16` | `vmovl_s8` |
| Dot accumulate | `_mm256_madd_epi16` | `vpaddlq_s16` |

## Exporting models

Use the `scripts/export_bin.py` script to convert PyTorch/JAX checkpoints:

```bash
python3 scripts/export_bin.py --version 1b --ckpt_path ./bitmamba_1b.msgpack --output_name bitmamba_1b.bin
python3 scripts/export_bin.py --version 250m --ckpt_path ./bitmamba_250m.msgpack --output_name bitmamba_250m.bin
```

## Related work

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024)
- [Aquantic Research — GPU-to-CPU/ARM Transposition](https://github.com/rasata/aquantic-research-gpu-to-cpu-transposition)

## License

MIT License (same as upstream)

## Credits

- Original BitMamba.cpp: [Zhayr1/bitmamba.cpp](https://github.com/Zhayr1/bitmamba.cpp)
- ARM NEON port: Aquantic Research
