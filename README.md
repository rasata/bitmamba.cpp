# BitMamba.cpp — Cross-Platform SSM Inference (ARM NEON / x86 AVX2 / AVX-512)

> **Fork of [Zhayr1/bitmamba.cpp](https://github.com/Zhayr1/bitmamba.cpp)** with ARM NEON, AVX-512, and scalar fallback support.

This fork ports the BitMamba-2 inference engine from x86 AVX2-only to a **cross-platform implementation** supporting x86 (AVX2, AVX-512), ARM (NEON), and scalar fallback, demonstrating the **O(1) memory property of State Space Models** on non-GPU architectures.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/Paper-Zenodo-00649C.svg)](https://doi.org/10.5281/zenodo.18394665)
[![Hugging Face 1B](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-1B%20Model-FFD21E)](https://huggingface.co/Zhayr1/BitMamba-2-1B)
[![Hugging Face 255M](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-255M%20Model-FFD21E)](https://huggingface.co/Zhayr1/BitMamba-2-0.25B)

## What's new in this fork

- **ARM NEON support**: Full port of AVX2 SIMD kernels to ARM NEON intrinsics
- **Apple Silicon compilation**: Builds and runs natively on Apple M1/M2/M3/M4
- **AVX-512 native kernels** (opt-in): 512-bit intrinsics for rms_norm, quantization, and ternary matmul
- **Scalar fallback**: Automatic fallback for pre-AVX2 CPUs (Ivy Bridge, Sandy Bridge)
- **Compile-time SIMD detection**: Guards on `__AVX2__`+`__FMA__` (not just `__x86_64__`)
- **Cross-platform dispatch**: AVX-512 (opt-in) > AVX2 > NEON > scalar
- **OpenMP on macOS**: CMake configured for AppleClang + Homebrew libomp
- **Benchmark data**: Cross-platform results across 4 CPUs and 4 SIMD levels

## Benchmark Results — Apple M1 (ARM NEON)

| Model | Weights | Speed | Latency/token | Prefill (8 tok) | Stable over length? |
|-------|---------|-------|---------------|-----------------|---------------------|
| BitMamba-2 **255M** | 246 MB | **82.5 tok/s** | 12.1 ms | 69.6 ms | Yes |
| BitMamba-2 **1B** | 614 MB | **27.9 tok/s** | 35.9 ms | 242 ms | Yes |

**Key finding**: Speed is **perfectly constant** regardless of sequence length (50, 200, or more tokens). This experimentally validates the **O(1) memory** property of SSM architectures — unlike Transformers whose memory grows with sequence length.

### Cross-platform comparison

| Model | Hardware | SIMD | tok/s |
|-------|----------|------|-------|
| BitMamba-2 255M | Intel i5-3230M (Ivy Bridge) | Scalar (SSE4.2) | 2.42 |
| BitMamba-2 1B | Intel i5-3230M (Ivy Bridge) | Scalar (SSE4.2) | 0.58 |
| **BitMamba-2 255M** | **Apple M1** | **NEON 128-bit** | **82.5** |
| **BitMamba-2 1B** | **Apple M1** | **NEON 128-bit** | **27.9** |
| BitMamba-2 255M | Xeon Silver 4210R | AVX2 path on AVX-512 | 113.1 |
| BitMamba-2 1B | Xeon Silver 4210R | AVX2 path on AVX-512 | 47.6 |
| BitMamba-2 255M | Xeon Silver 4210R | **AVX-512 native** | 92.4 |
| BitMamba-2 1B | Xeon Silver 4210R | **AVX-512 native** | 32.6 |
| BitMamba-2 255M | Intel i3-12100F | AVX2 (ref, upstream) | ~146 |
| BitMamba-2 1B | Intel i3-12100F | AVX2 (ref, upstream) | ~53 |

> **AVX-512 note**: Native AVX-512 kernels are 18-31% *slower* than AVX2 on Cascade Lake due to frequency throttling. AVX-512 is disabled by default (opt-in via `-DUSE_AVX512=ON`). It may be beneficial on Ice Lake+ or AMD Zen 4/5 where throttling is reduced.

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
| x86_64 (Intel/AMD) | AVX2 + FMA | Supported (original + improved guards) |
| x86_64 (Intel Xeon) | AVX-512 | Supported (opt-in via `-DUSE_AVX512=ON`) |
| x86_64 (pre-Haswell) | Scalar (SSE4.2) | Supported (automatic fallback) |
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

# Linux x86 (AVX2, default — works on AVX-512 hardware too)
cmake -B build
cmake --build build

# Linux x86 with AVX-512 native kernels (opt-in, not recommended on Cascade Lake)
cmake -B build -DUSE_AVX512=ON
cmake --build build

# Linux x86 pre-AVX2 (scalar fallback, automatic)
cmake -B build
cmake --build build  # auto-detects missing AVX2, uses scalar path
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

**`src/kernels.cpp`** — Core SIMD kernels with 4-level dispatch:

```cpp
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(BITMAMBA_USE_AVX512)
    #define BITMAMBA_AVX512   // 512-bit: 16 float / 64 int8 per iter
    #define BITMAMBA_X86
#elif defined(__AVX2__) && defined(__FMA__)
    #define BITMAMBA_X86      // 256-bit: 8 float / 32 int8 per iter
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #define BITMAMBA_ARM      // 128-bit: 4 float / 16 int8 per iter
#else
    #define BITMAMBA_SCALAR   // Pure scalar fallback
#endif
```

**`CMakeLists.txt`** — Platform-adaptive build:
- Apple: `-mcpu=native -Xclang -fopenmp` + Homebrew libomp
- Linux: `-march=native -fopenmp`
- AVX-512: opt-in via `-DUSE_AVX512=ON` (adds `-DBITMAMBA_USE_AVX512`)

**`examples/main.cpp`** — Removed unused `#include <immintrin.h>`

### SIMD kernel mapping

| Operation | AVX-512 (opt-in) | AVX2 (x86) | NEON (ARM) |
|-----------|-----------------|-----------|------------|
| Float32 FMA | `_mm512_fmadd_ps` (16-wide) | `_mm256_fmadd_ps` (8-wide) | `vfmaq_f32` (4-wide) |
| Horizontal sum | `_mm512_reduce_add_ps` | manual 8-elem | `vaddvq_f32` |
| Int8 load | `_mm512_loadu_si512` (64B) | `_mm256_loadu_si256` (32B) | `vld1q_s8` (16B) |
| Ternary matmul | mask compare + blend | `_mm256_sign_epi8` | mask + negate + add |
| Widen int8->int16 | `_mm512_cvtepi8_epi16` | `_mm256_cvtepi8_epi16` | `vmovl_s8` |
| Dot accumulate | `_mm512_madd_epi16` | `_mm256_madd_epi16` | `vpaddlq_s16` |
| Quantize pack | `_mm512_cvtepi32_epi16` + `_mm256_cvtepi16_epi8` | manual float->round | `vcvtnq_s32_f32` + narrow |

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
- ARM NEON port, AVX-512 kernels, scalar fallback, cross-platform benchmarks: Aquantic Research
