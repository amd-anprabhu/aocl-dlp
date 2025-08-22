/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "utils/conversion_utils.hh"
#include <cstdint>

namespace dlp { namespace testing { namespace utils {

    float bf16_to_f32(bfloat16 bf16_val)
    {
        union
        {
            float    f;
            uint32_t u;
        } float_bits;
        float_bits.u = static_cast<uint32_t>(static_cast<uint16_t>(bf16_val))
                       << 16U;
        return float_bits.f;
    }

    bfloat16 f32_to_bf16(float f32_val)
    {
        union
        {
            float    f;
            uint32_t u;
        } bits;
        bits.f = f32_val;
        // Extract LSB of BF16 part for ties-to-even
        uint32_t lsb = (bits.u >> 16U) & 1U;
        uint32_t rounding_bias =
            0x7FFFU + lsb; // 0x7FFF for round, +lsb for ties-to-even
        uint32_t rounded    = bits.u + rounding_bias;
        uint16_t bf16_upper = static_cast<uint16_t>(rounded >> 16U);
        return static_cast<bfloat16>(bf16_upper);
    }

}}} // namespace dlp::testing::utils
