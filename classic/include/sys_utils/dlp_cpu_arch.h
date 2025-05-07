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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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
 *
 */

#ifndef DLP_CPU_ARCH_H
#define DLP_CPU_ARCH_H

#include "classic/dlp_base_types.h"

typedef enum
{
    DLP_ARCH_ERROR = 0,
    DLP_ARCH_GENERIC,

    // AMD
    DLP_ARCH_ZEN5,
    DLP_ARCH_ZEN4,
    DLP_ARCH_ZEN3,
    DLP_ARCH_ZEN2,
    DLP_ARCH_ZEN,

    DLP_NUM_ARCHS
} dlp_arch_t;

dlp_arch_t
dlp_get_arch(void);

// API to check if AVX2 and FMA3 are supported or not on the current platform.
bool
dlp_cpuid_is_avx2fma3_supported(void);

// API to check if AVX512 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512_supported(void);

// API to check if AVX512_VNNI is supported or not on the current platform.
bool
dlp_cpuid_is_avx512vnni_supported(void);

// API to check if AVX512_bf16 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512bf16_supported(void);

uint32_t
dlp_cpuid_query_fp_datapath(void);

#endif // DLP_CPU_ARCH_H
