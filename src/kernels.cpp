#include "bitmamba/kernels.h"
#include "bitmamba/quantization.h"
#include <algorithm>
#include <cmath>

// =========================================================================
// Platform dispatch: AVX-512 > AVX2 > NEON > scalar
// =========================================================================
// Note: AVX-512 on Cascade Lake (model 85) triggers aggressive frequency
// throttling (~20-30%) that negates the 2x SIMD width gain. AVX-512
// kernels are only enabled when BITMAMBA_FORCE_AVX512 is defined at
// compile time, or via the CMake option -DUSE_AVX512=ON.
// By default, AVX2 is used even on AVX-512 capable hardware.
// =========================================================================

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__)) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(BITMAMBA_USE_AVX512)
    #define BITMAMBA_AVX512
    #define BITMAMBA_X86
    #include <immintrin.h>
#elif (defined(__x86_64__) || defined(_M_X64) || defined(__i386__)) && defined(__AVX2__) && defined(__FMA__)
    #define BITMAMBA_X86
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #define BITMAMBA_ARM
    #include <arm_neon.h>
#else
    #define BITMAMBA_SCALAR
#endif

namespace bitmamba {

// =========================================================================
// rms_norm
// =========================================================================

void rms_norm(const std::vector<float>& x, const Tensor& weight, std::vector<float>& out) {
    float sum_sq = 0.0f;
    int size = x.size();
    int i = 0;

#if defined(BITMAMBA_AVX512)
    __m512 sum_vec512 = _mm512_setzero_ps();
    for (; i <= size - 16; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        sum_vec512 = _mm512_fmadd_ps(v, v, sum_vec512);
    }
    sum_sq += _mm512_reduce_add_ps(sum_vec512);

#elif defined(BITMAMBA_X86)
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i <= size - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
    }
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    for (int k = 0; k < 8; ++k) sum_sq += temp[k];

#elif defined(BITMAMBA_ARM)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i <= size - 4; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        sum_vec = vfmaq_f32(sum_vec, v, v);
    }
    sum_sq += vaddvq_f32(sum_vec);
#endif

    // Scalar tail
    for (; i < size; ++i) sum_sq += x[i] * x[i];

    float rms = 1.0f / sqrtf(sum_sq / size + 1e-6f);

    i = 0;
#if defined(BITMAMBA_AVX512)
    __m512 rms_vec512 = _mm512_set1_ps(rms);
    for (i = 0; i <= size - 16; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vw = _mm512_loadu_ps(&weight.data[i]);
        _mm512_storeu_ps(&out[i], _mm512_mul_ps(_mm512_mul_ps(vx, rms_vec512), vw));
    }

#elif defined(BITMAMBA_X86)
    __m256 rms_vec = _mm256_set1_ps(rms);
    for (i = 0; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vw = _mm256_loadu_ps(&weight.data[i]);
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_mul_ps(vx, rms_vec), vw));
    }

#elif defined(BITMAMBA_ARM)
    float32x4_t rms_vec = vdupq_n_f32(rms);
    for (i = 0; i <= size - 4; i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        float32x4_t vw = vld1q_f32(&weight.data[i]);
        vst1q_f32(&out[i], vmulq_f32(vmulq_f32(vx, rms_vec), vw));
    }
#endif

    // Scalar tail
    for (; i < size; ++i) out[i] = x[i] * rms * weight.data[i];
}

// =========================================================================
// bitlinear_forward — the core ternary matmul kernel
// =========================================================================

void bitlinear_forward(const std::vector<float>& x, const Tensor& w, const Tensor& norm_w, std::vector<float>& out) {
    int n = x.size();
    std::vector<float> x_norm(n);
    rms_norm(x, norm_w, x_norm);

    // Quantize activations to INT8
    float max_abs = 0.0f;
    for (float v : x_norm) max_abs = std::max(max_abs, std::abs(v));
    float scale_x = 127.0f / (max_abs + 1e-5f);

    std::vector<int8_t> x_quant(n + 64, 0);

    int i = 0;
#if defined(BITMAMBA_AVX512)
    __m512 scale_v512 = _mm512_set1_ps(scale_x);
    __m512 min_v512 = _mm512_set1_ps(-128.0f);
    __m512 max_v512 = _mm512_set1_ps(127.0f);
    for (; i <= n - 16; i += 16) {
        __m512 v = _mm512_loadu_ps(&x_norm[i]);
        v = _mm512_mul_ps(v, scale_v512);
        v = _mm512_max_ps(min_v512, _mm512_min_ps(v, max_v512));
        __m512i vi = _mm512_cvt_roundps_epi32(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Pack int32 -> int16 -> int8: 16 values
        __m256i vi16 = _mm512_cvtepi32_epi16(vi);
        __m128i vi8 = _mm256_cvtepi16_epi8(vi16);
        _mm_storeu_si128((__m128i*)&x_quant[i], vi8);
    }

#elif defined(BITMAMBA_X86)
    __m256 scale_v = _mm256_set1_ps(scale_x);
    __m256 min_v = _mm256_set1_ps(-128.0f);
    __m256 max_v = _mm256_set1_ps(127.0f);
    for (; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(&x_norm[i]);
        v = _mm256_mul_ps(v, scale_v);
        v = _mm256_max_ps(min_v, _mm256_min_ps(v, max_v));
        float buf[8]; _mm256_storeu_ps(buf, v);
        for (int k = 0; k < 8; ++k) x_quant[i+k] = (int8_t)roundf(buf[k]);
    }

#elif defined(BITMAMBA_ARM)
    float32x4_t scale_v = vdupq_n_f32(scale_x);
    float32x4_t min_v = vdupq_n_f32(-128.0f);
    float32x4_t max_v = vdupq_n_f32(127.0f);
    for (; i <= n - 4; i += 4) {
        float32x4_t v = vld1q_f32(&x_norm[i]);
        v = vmulq_f32(v, scale_v);
        v = vmaxq_f32(min_v, vminq_f32(v, max_v));
        int32x4_t vi = vcvtnq_s32_f32(v);
        int16x4_t vi16 = vmovn_s32(vi);
        int8x8_t vi8 = vmovn_s16(vcombine_s16(vi16, vi16));
        vst1_lane_s8(&x_quant[i], vi8, 0);
        vst1_lane_s8(&x_quant[i+1], vi8, 1);
        vst1_lane_s8(&x_quant[i+2], vi8, 2);
        vst1_lane_s8(&x_quant[i+3], vi8, 3);
    }
#endif

    // Scalar tail
    for (; i < n; ++i) {
        float val = x_norm[i] * scale_x;
        x_quant[i] = (int8_t)(val > 127.f ? 127 : (val < -128.f ? -128 : (int8_t)roundf(val)));
    }

    // Ternary matmul: W (packed 2-bit) x x_quant (INT8)
    int cols = w.cols;
    int packed_stride = (cols + 3) / 4;
    const uint8_t* packed_ptr = w.packed_data.data();

    #pragma omp parallel for
    for (int r = 0; r < w.rows; ++r) {
        int row_offset = r * packed_stride;
        int32_t total = 0;
        int c = 0;

#if defined(BITMAMBA_AVX512)
        // AVX-512: process 64 int8 elements per iteration (512-bit)
        __m512i acc_vec512 = _mm512_setzero_si512();

        for (; c <= cols - 64; c += 64) {
            const uint8_t* p = packed_ptr + row_offset + c/4;

            // Unpack 16 bytes = 64 ternary weights via LUT
            alignas(64) int8_t w_temp[64];
            uint32_t* w_ptr32 = (uint32_t*)w_temp;
            for (int k = 0; k < 16; ++k)
                w_ptr32[k] = UNPACK_LUT[p[k]];

            __m512i w_vec = _mm512_load_si512((__m512i*)w_temp);
            __m512i x_vec = _mm512_loadu_si512((__m512i*)&x_quant[c]);

            // Ternary multiply using sign trick:
            // For w in {-1, 0, +1}: result = x * sign(w), zero where w=0
            // _mm512_sign_epi8 does not exist in AVX-512.
            // Strategy: separate positive/negative masks, blend results.
            __m512i zero = _mm512_setzero_si512();
            __m512i neg_x = _mm512_sub_epi8(zero, x_vec);

            // mask where w == +1 (0x01)
            __mmask64 pos_mask = _mm512_cmpeq_epi8_mask(w_vec, _mm512_set1_epi8(1));
            // mask where w == -1 (0xFF)
            __mmask64 neg_mask = _mm512_cmpeq_epi8_mask(w_vec, _mm512_set1_epi8(-1));

            // Blend: pick x where w=+1, -x where w=-1, 0 elsewhere
            __m512i prod = _mm512_mask_mov_epi8(zero, pos_mask, x_vec);
            prod = _mm512_mask_mov_epi8(prod, neg_mask, neg_x);

            // Widen int8 -> int16 and horizontal pairwise sum to int32
            // Split 512-bit into two 256-bit halves
            __m256i prod_lo256 = _mm512_castsi512_si256(prod);
            __m256i prod_hi256 = _mm512_extracti64x4_epi64(prod, 1);

            // int8 -> int16 sign-extend (256->512)
            __m512i prod_lo16 = _mm512_cvtepi8_epi16(prod_lo256);
            __m512i prod_hi16 = _mm512_cvtepi8_epi16(prod_hi256);

            // madd: pairs of int16 -> int32 with factor 1
            __m512i ones_16_512 = _mm512_set1_epi16(1);
            acc_vec512 = _mm512_add_epi32(acc_vec512, _mm512_madd_epi16(prod_lo16, ones_16_512));
            acc_vec512 = _mm512_add_epi32(acc_vec512, _mm512_madd_epi16(prod_hi16, ones_16_512));
        }

        total += _mm512_reduce_add_epi32(acc_vec512);

        // AVX2 tail for remaining 32-element chunks
        __m256i ones_16 = _mm256_set1_epi16(1);
        __m256i acc_vec = _mm256_setzero_si256();

        for (; c <= cols - 32; c += 32) {
            const uint8_t* p = packed_ptr + row_offset + c/4;
            alignas(32) int8_t w_temp[32];
            uint32_t* w_ptr32 = (uint32_t*)w_temp;
            for (int k = 0; k < 8; ++k)
                w_ptr32[k] = UNPACK_LUT[p[k]];

            __m256i w_vec = _mm256_load_si256((__m256i*)w_temp);
            __m256i x_vec = _mm256_loadu_si256((__m256i*)&x_quant[c]);
            __m256i prod = _mm256_sign_epi8(x_vec, w_vec);
            __m256i prod_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod));
            __m256i prod_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod, 1));
            acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(prod_lo, ones_16));
            acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(prod_hi, ones_16));
        }

        int32_t temp_acc[8];
        _mm256_storeu_si256((__m256i*)temp_acc, acc_vec);
        for (int k = 0; k < 8; ++k) total += temp_acc[k];

#elif defined(BITMAMBA_X86)
        __m256i ones_16 = _mm256_set1_epi16(1);
        __m256i acc_vec = _mm256_setzero_si256();

        for (; c <= cols - 32; c += 32) {
            const uint8_t* p = packed_ptr + row_offset + c/4;
            alignas(32) int8_t w_temp[32];
            uint32_t* w_ptr32 = (uint32_t*)w_temp;
            w_ptr32[0] = UNPACK_LUT[p[0]];
            w_ptr32[1] = UNPACK_LUT[p[1]];
            w_ptr32[2] = UNPACK_LUT[p[2]];
            w_ptr32[3] = UNPACK_LUT[p[3]];
            w_ptr32[4] = UNPACK_LUT[p[4]];
            w_ptr32[5] = UNPACK_LUT[p[5]];
            w_ptr32[6] = UNPACK_LUT[p[6]];
            w_ptr32[7] = UNPACK_LUT[p[7]];

            __m256i w_vec = _mm256_load_si256((__m256i*)w_temp);
            __m256i x_vec = _mm256_loadu_si256((__m256i*)&x_quant[c]);
            __m256i prod = _mm256_sign_epi8(x_vec, w_vec);
            __m256i prod_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod));
            __m256i prod_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod, 1));
            acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(prod_lo, ones_16));
            acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(prod_hi, ones_16));
        }

        int32_t temp_acc[8];
        _mm256_storeu_si256((__m256i*)temp_acc, acc_vec);
        for (int k = 0; k < 8; ++k) total += temp_acc[k];

#elif defined(BITMAMBA_ARM)
        int32x4_t acc_vec = vdupq_n_s32(0);

        for (; c <= cols - 16; c += 16) {
            const uint8_t* p = packed_ptr + row_offset + c/4;

            // Unpack 4 bytes = 16 ternary weights using LUT
            alignas(16) int8_t w_temp[16];
            uint32_t* w_ptr32 = (uint32_t*)w_temp;
            w_ptr32[0] = UNPACK_LUT[p[0]];
            w_ptr32[1] = UNPACK_LUT[p[1]];
            w_ptr32[2] = UNPACK_LUT[p[2]];
            w_ptr32[3] = UNPACK_LUT[p[3]];

            int8x16_t w_vec = vld1q_s8(w_temp);
            int8x16_t x_vec = vld1q_s8(&x_quant[c]);

            // Ternary multiply: w in {-1, 0, +1}
            // result = x * sign(w), zero where w=0
            // Use: negate where w=-1, keep where w=+1, zero where w=0
            int8x16_t pos_mask = vceqq_s8(w_vec, vdupq_n_s8(1));
            int8x16_t neg_mask = vceqq_s8(w_vec, vdupq_n_s8(-1));
            int8x16_t pos_contrib = vandq_s8(x_vec, pos_mask);
            int8x16_t neg_contrib = vandq_s8(vnegq_s8(x_vec), neg_mask);
            int8x16_t prod = vaddq_s8(pos_contrib, neg_contrib);

            // Widen and accumulate: int8 -> int16 -> int32
            int16x8_t prod_lo = vmovl_s8(vget_low_s8(prod));
            int16x8_t prod_hi = vmovl_s8(vget_high_s8(prod));
            acc_vec = vaddq_s32(acc_vec, vpaddlq_s16(prod_lo));
            acc_vec = vaddq_s32(acc_vec, vpaddlq_s16(prod_hi));
        }

        total += vaddvq_s32(acc_vec);
#endif

        // Scalar tail
        for (; c < cols; ++c) {
            int byte_idx = c / 4;
            int bit_shift = (c % 4) * 2;
            int8_t w_val = ((packed_ptr[row_offset + byte_idx] >> bit_shift) & 0x03) - 1;
            if (w_val) total += (w_val == 1) ? x_quant[c] : -x_quant[c];
        }

        out[r] = (float)total / (scale_x * w.scale);
    }
}

}
