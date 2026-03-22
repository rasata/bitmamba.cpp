#include "bitmamba/kernels.h"
#include "bitmamba/quantization.h"
#include <algorithm>
#include <cmath>

// =========================================================================
// Platform dispatch: AVX2 (x86) or NEON (ARM/Apple Silicon)
// =========================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
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

#if defined(BITMAMBA_X86)
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
#if defined(BITMAMBA_X86)
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

    std::vector<int8_t> x_quant(n + 32, 0);

    int i = 0;
#if defined(BITMAMBA_X86)
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

#if defined(BITMAMBA_X86)
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
