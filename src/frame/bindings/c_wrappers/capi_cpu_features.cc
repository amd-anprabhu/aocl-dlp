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
 *
 */

#include "bindings/c_wrappers/capi_cpu_features.h"
#include "cpu_utils/cpu_features.hh"

using namespace dlp::cpu_utils;

// Determine if the CPU has support for AVX2 and FMA3.
bool
dlp_cpuid_is_avx2fma3_supported(void)
{
    static const bool is_avx2fma3_supported = [&]() -> bool {
        std::vector<isaFeature> reqFeatures{ isaFeature::avx, isaFeature::fma3,
                                             isaFeature::avx2 };
        return cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();

    return is_avx2fma3_supported;
}

// Determine if the CPU has support for AVX512.
bool
dlp_cpuid_is_avx512_supported(void)
{
    static const bool is_avx512_supported = [&]() -> bool {
        std::vector<isaFeature> reqFeatures{
            isaFeature::avx,      isaFeature::fma3,     isaFeature::avx2,
            isaFeature::avx512f,  isaFeature::avx512dq, isaFeature::avx512cd,
            isaFeature::avx512bw, isaFeature::avx512vl
        };
        return cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();

    return is_avx512_supported;
}

// Determine if the CPU has support for AVX512_VNNI.
bool
dlp_cpuid_is_avx512vnni_supported(void)
{
    static const bool is_avx512vnni_supported = [&]() -> bool {
        std::vector<isaFeature> reqFeatures{
            isaFeature::avx,      isaFeature::fma3,     isaFeature::avx2,
            isaFeature::avx512f,  isaFeature::avx512dq, isaFeature::avx512cd,
            isaFeature::avx512bw, isaFeature::avx512vl, isaFeature::avx512vnni
        };
        return cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();

    return is_avx512vnni_supported;
}

// Determine if the CPU has support for AVX512_BF16.
bool
dlp_cpuid_is_avx512bf16_supported(void)
{
    static const bool is_avx512bf16_supported = [&]() -> bool {
        std::vector<isaFeature> reqFeatures{
            isaFeature::avx,       isaFeature::fma3,     isaFeature::avx2,
            isaFeature::avx512f,   isaFeature::avx512dq, isaFeature::avx512cd,
            isaFeature::avx512bw,  isaFeature::avx512vl, isaFeature::avx512vnni,
            isaFeature::avx512bf16
        };
        return cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();

    return is_avx512bf16_supported;
}

uint32_t
dlp_cpuid_query_fp_datapath(void)
{
    static const uint32_t fp_datapath = [&]() -> uint32_t {
        if (cpuFeaturesInstance().getCpuVendor() == cpuVendor::amd) {
            if (cpuFeaturesInstance().hasFeature(isaFeature::datapath_fp512)) {
                return DATAPATH_FP512;
            }
            if (cpuFeaturesInstance().hasFeature(isaFeature::datapath_fp256)) {
                return DATAPATH_FP256;
            }
            if (cpuFeaturesInstance().hasFeature(isaFeature::datapath_fp128)) {
                return DATAPATH_FP128;
            }
        }
        return DATAPATH_INVALID;
    }();

    return fp_datapath;
}

bool
dlp_cpuid_is_similar_zen5_arch()
{
    static const bool is_zen5 = [&]() -> bool {
        std::vector<isaFeature> reqFeatures{ isaFeature::sse3,
                                             isaFeature::ssse3,
                                             isaFeature::sse41,
                                             isaFeature::sse42,
                                             isaFeature::avx,
                                             isaFeature::fma3,
                                             isaFeature::avx2,
                                             isaFeature::avx512f,
                                             isaFeature::avx512dq,
                                             isaFeature::avx512cd,
                                             isaFeature::avx512bw,
                                             isaFeature::avx512vl,
                                             isaFeature::avx512vnni,
                                             isaFeature::avx512bf16,
                                             isaFeature::movdiri,
                                             isaFeature::movdir64b,
                                             isaFeature::avx512vp2intersect,
                                             isaFeature::avxvnni };
        return cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();

    return is_zen5;
}

bool
dlp_cpuid_is_similar_zen4_arch()
{
    static const bool is_zen4 = [&]() -> bool {
        std::vector<isaFeature> reqFeatures{
            isaFeature::sse3,       isaFeature::ssse3,     isaFeature::sse41,
            isaFeature::sse42,      isaFeature::avx,       isaFeature::fma3,
            isaFeature::avx2,       isaFeature::avx512f,   isaFeature::avx512dq,
            isaFeature::avx512cd,   isaFeature::avx512bw,  isaFeature::avx512vl,
            isaFeature::avx512vnni, isaFeature::avx512bf16
        };
        return cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();

    return is_zen4;
}

bool
dlp_cpuid_is_similar_zen_arch()
{
    static const bool is_zen = [&]() -> bool {
        std::vector<isaFeature> reqFeatures{ isaFeature::avx, isaFeature::fma3,
                                             isaFeature::avx2 };
        return cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();

    return is_zen;
}

static dlp_arch_t
_dlp_cpuid_query_arch_id()
{
    dlp_arch_t archId = DLP_ARCH_GENERIC;

    auto vendor = cpuFeaturesInstance().getCpuVendor();
    if (vendor == cpuVendor::intel) {
        // Check for each Intel configuration that is enabled, check for that
        // microarchitecture. We check from most recent to most dated.
#ifdef DLP_CONFIG_ZEN4
        // Even if not optimized for Intel processors, this should
        // generally perform better than skx codepath.
        // Currently only enabled for zen4 and amdzen configurations
        if (dlp_cpuid_is_avx512_supported()) {
            archId = DLP_ARCH_ZEN4;
        }
#endif
#ifdef DLP_CONFIG_ZEN3
        // Even if not optimized for Intel processors, this should
        // generally perform better than haswell codepath.
        // Currently only enabled for zen3 and amdzen configurations
        if (dlp_cpuid_is_avx2fma3_supported()) {
            archId = DLP_ARCH_ZEN3;
        }
#endif
    } else if (vendor == cpuVendor::amd) {
        // The ARCH is decided based on the dlp config set during compile
        // time and the ISA supported. The model and family id is NOT used
        // for determining the same.
#ifdef DLP_CONFIG_ZEN5
        if (dlp_cpuid_is_similar_zen5_arch()) {
            archId = DLP_ARCH_ZEN5;
        }
#endif
#ifdef DLP_CONFIG_ZEN4
        if (dlp_cpuid_is_similar_zen4_arch()) {
            archId = DLP_ARCH_ZEN4;
        }
#endif
#ifdef DLP_CONFIG_ZEN5
        // Fallback test for future AMD processors
        // Assume zen5 (if available) is preferable to zen4.
        if (dlp_cpuid_is_avx512bf16_supported()) {
            archId = DLP_ARCH_ZEN5;
        }
#endif
#ifdef DLP_CONFIG_ZEN4
        // Fallback test for future AMD processors
        // Use zen4 if zen5 is not available.
        if (dlp_cpuid_is_avx512bf16_supported()) {
            archId = DLP_ARCH_ZEN4;
        }
#endif
#ifdef DLP_CONFIG_ZEN3
        if (dlp_cpuid_is_similar_zen_arch()) {
            archId = DLP_ARCH_ZEN3;
        }
#endif
#ifdef DLP_CONFIG_ZEN2
        if (dlp_cpuid_is_similar_zen_arch()) {
            archId = DLP_ARCH_ZEN2;
        }
#endif
#ifdef DLP_CONFIG_ZEN
        if (dlp_cpuid_is_similar_zen_arch()) {
            archId = DLP_ARCH_ZEN;
        }
#endif
#ifdef DLP_CONFIG_ZEN3
        // Fallback test for future AMD processors
        // Use zen3 if AVX512 support is not available but AVX2 is.
        if (dlp_cpuid_is_avx2fma3_supported()) {
            archId = DLP_ARCH_ZEN3;
        }
#endif
    }

    return archId;
}

dlp_arch_t
dlp_cpuid_query_arch_id(void)
{
    static const dlp_arch_t archId = _dlp_cpuid_query_arch_id();
    return archId;
}
