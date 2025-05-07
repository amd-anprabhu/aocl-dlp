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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "classic/aocl_lib_interface_apis.h"
#include "sys_utils/dlp_cpu_arch.h"
#include "sys_utils/lpgemm_sys.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386)                  \
    || defined(_M_IX86)

#include "cpuid.h"

enum
{
    // input register(s)     output register
    FEATURE_MASK_SSE3       = (1u << 0),  // cpuid[eax=1]          :ecx[0]
    FEATURE_MASK_SSSE3      = (1u << 9),  // cpuid[eax=1]          :ecx[9]
    FEATURE_MASK_SSE41      = (1u << 19), // cpuid[eax=1]          :ecx[19]
    FEATURE_MASK_SSE42      = (1u << 20), // cpuid[eax=1]          :ecx[20]
    FEATURE_MASK_AVX        = (1u << 28), // cpuid[eax=1]          :ecx[28]
    FEATURE_MASK_AVX2       = (1u << 5),  // cpuid[eax=7,ecx=0]    :ebx[5]
    FEATURE_MASK_FMA3       = (1u << 12), // cpuid[eax=1]          :ecx[12]
    FEATURE_MASK_FMA4       = (1u << 16), // cpuid[eax=0x80000001] :ecx[16]
    FEATURE_MASK_AVX512F    = (1u << 16), // cpuid[eax=7,ecx=0]    :ebx[16]
    FEATURE_MASK_AVX512DQ   = (1u << 17), // cpuid[eax=7,ecx=0]    :ebx[17]
    FEATURE_MASK_AVX512PF   = (1u << 26), // cpuid[eax=7,ecx=0]    :ebx[26]
    FEATURE_MASK_AVX512ER   = (1u << 27), // cpuid[eax=7,ecx=0]    :ebx[27]
    FEATURE_MASK_AVX512CD   = (1u << 28), // cpuid[eax=7,ecx=0]    :ebx[28]
    FEATURE_MASK_AVX512BW   = (1u << 30), // cpuid[eax=7,ecx=0]    :ebx[30]
    FEATURE_MASK_AVX512VL   = (1u << 31), // cpuid[eax=7,ecx=0]    :ebx[31]
    FEATURE_MASK_AVX512VNNI = (1u << 11), // cpuid[eax=7,ecx=0]    :ecx[11]
    FEATURE_MASK_MOVDIRI    = (1u << 27), // cpuid[eax=7,ecx=0]    :ecx[27]
    FEATURE_MASK_MOVDIR64B  = (1u << 28), // cpuid[eax=7,ecx=0]    :ecx[28]
    FEATURE_MASK_AVX512VP2INTERSECT = (1u << 8), // cpuid[eax=7,ecx=0] :edx[8]
    FEATURE_MASK_AVXVNNI    = (1u << 4), // cpuid[eax=7,ecx=1]    :eax[4]
    FEATURE_MASK_AVX512BF16 = (1u << 5), // cpuid[eax=7,ecx=1]    :eax[5]
    FEATURE_MASK_XGETBV = (1u << 26) | (1u << 27), // cpuid[eax=1] :ecx[27:26]
    XGETBV_MASK_XMM     = 0x02u,                   // xcr0[1]
    XGETBV_MASK_YMM     = 0x04u,                   // xcr0[2]
    XGETBV_MASK_ZMM     = 0xe0u,                   // xcr0[7:5]
    FEATURE_MASK_DATAPATH_FP128 = (1u << 0), // cpuid[eax=0x8000001A] :eax[0]
    FEATURE_MASK_DATAPATH_FP256 = (1u << 2), // cpuid[eax=0x8000001A] :eax[2]
    FEATURE_MASK_DATAPATH_FP512 = (1u << 3)  // cpuid[eax=0x8000001A] :eax[3]
};

enum
{
    VENDOR_INTEL = 0,
    VENDOR_AMD,
    VENDOR_UNKNOWN
};

enum
{
    FEATURE_SSE3               = 0x0001,
    FEATURE_SSSE3              = 0x0002,
    FEATURE_SSE41              = 0x0004,
    FEATURE_SSE42              = 0x0008,
    FEATURE_AVX                = 0x0010,
    FEATURE_AVX2               = 0x0020,
    FEATURE_FMA3               = 0x0040,
    FEATURE_FMA4               = 0x0080,
    FEATURE_AVX512F            = 0x0100,
    FEATURE_AVX512DQ           = 0x0200,
    FEATURE_AVX512PF           = 0x0400,
    FEATURE_AVX512ER           = 0x0800,
    FEATURE_AVX512CD           = 0x1000,
    FEATURE_AVX512BW           = 0x2000,
    FEATURE_AVX512VL           = 0x4000,
    FEATURE_AVX512VNNI         = 0x8000,
    FEATURE_AVX512BF16         = 0x10000,
    FEATURE_AVXVNNI            = 0x20000,
    FEATURE_AVX512VP2INTERSECT = 0x40000,
    FEATURE_MOVDIRI            = 0x80000,
    FEATURE_MOVDIR64B          = 0x100000,
    FEATURE_DATAPATH_FP128     = 0x200000,
    FEATURE_DATAPATH_FP256     = 0x400000,
    FEATURE_DATAPATH_FP512     = 0x800000
};

// To reduce confusion, include MOVU bit so enum values match those in
// CPUID_Fn8000001A_EAX id function.
enum
{
    DATAPATH_UNSET = -1,
    DATAPATH_FP128,
    DATAPATH_MOVU,
    DATAPATH_FP256,
    DATAPATH_FP512
};

static bool               aocl_e_i       = FALSE;
static dlp_arch_t         arch_id        = -1;
static dlp_arch_t         actual_arch_id = -1;
static dlp_pthread_once_t once_id_init   = DLP_PTHREAD_ONCE_INIT;
static dlp_pthread_once_t once_id_check  = DLP_PTHREAD_ONCE_INIT;

// Variables for denoting if an ISA is supported or not.
static bool is_avx2fma3_supported   = FALSE;
static bool is_avx512_supported     = FALSE;
static bool is_avx512vnni_supported = FALSE;
static bool is_avx512bf16_supported = FALSE;

// Variable to represent FP/SIMD execution datapath width.
static uint32_t dlp_fp_datapath = -1;

DLP_INLINE bool
dlp_cpuid_has_features(uint32_t have, uint32_t want)
{
    return (have & want) == want;
}

// Determine if the CPU has support for AVX2 and FMA3.
void
dlp_cpuid_check_avx2fma3_support(uint32_t features)
{
    // Check for expected CPU features.
    const uint32_t expected = FEATURE_AVX | FEATURE_FMA3 | FEATURE_AVX2;

    if (!dlp_cpuid_has_features(features, expected)) {
        is_avx2fma3_supported = FALSE;
    } else {
        is_avx2fma3_supported = TRUE;
    }
}

// Determine if the CPU has support for AVX512.
void
dlp_cpuid_check_avx512_support(uint32_t features)
{
    // Check for expected CPU features.
    const uint32_t expected = FEATURE_AVX | FEATURE_FMA3 | FEATURE_AVX2
                              | FEATURE_AVX512F | FEATURE_AVX512DQ
                              | FEATURE_AVX512CD | FEATURE_AVX512BW
                              | FEATURE_AVX512VL;

    if (!dlp_cpuid_has_features(features, expected)) {
        is_avx512_supported = FALSE;
    } else {
        is_avx512_supported = TRUE;
    }
}

// Determine if the CPU has support for AVX512_VNNI.
void
dlp_cpuid_check_avx512vnni_support(uint32_t features)
{
    // Check for expected CPU features.
    const uint32_t expected = FEATURE_AVX | FEATURE_FMA3 | FEATURE_AVX2
                              | FEATURE_AVX512F | FEATURE_AVX512DQ
                              | FEATURE_AVX512CD | FEATURE_AVX512BW
                              | FEATURE_AVX512VL | FEATURE_AVX512VNNI;

    if (!dlp_cpuid_has_features(features, expected)) {
        is_avx512vnni_supported = FALSE;
    } else {
        is_avx512vnni_supported = TRUE;
    }
}

// Determine if the CPU has support for AVX512_BF16.
void
dlp_cpuid_check_avx512bf16_support(uint32_t features)
{
    // Check for expected CPU features.
    const uint32_t expected =
        FEATURE_AVX | FEATURE_FMA3 | FEATURE_AVX2 | FEATURE_AVX512F
        | FEATURE_AVX512DQ | FEATURE_AVX512CD | FEATURE_AVX512BW
        | FEATURE_AVX512VL | FEATURE_AVX512VNNI | FEATURE_AVX512BF16;

    if (!dlp_cpuid_has_features(features, expected)) {
        is_avx512bf16_supported = FALSE;
    } else {
        is_avx512bf16_supported = TRUE;
    }
}

void
dlp_cpuid_check_datapath(uint32_t vendor, uint32_t features)
{
    if (vendor == VENDOR_AMD) {
        uint32_t expected;
        expected = FEATURE_DATAPATH_FP512;
        if (dlp_cpuid_has_features(features, expected)) {
            dlp_fp_datapath = DATAPATH_FP512;
            return;
        }
        expected = FEATURE_DATAPATH_FP256;
        if (dlp_cpuid_has_features(features, expected)) {
            dlp_fp_datapath = DATAPATH_FP256;
            return;
        }
        expected = FEATURE_DATAPATH_FP128;
        if (dlp_cpuid_has_features(features, expected)) {
            dlp_fp_datapath = DATAPATH_FP128;
            return;
        }
    }
}

uint32_t
dlp_cpuid_query(uint32_t* features)
{
    uint32_t eax, ebx, ecx, edx;

    *features = 0;

    uint32_t cpuid_max     = __get_cpuid_max(0, 0);
    uint32_t cpuid_max_ext = __get_cpuid_max(0x80000000u, 0);

    if (cpuid_max < 1)
        return VENDOR_UNKNOWN;

    // The fourth '0' serves as the NULL-terminator for the vendor string.
    uint32_t vendor_string[4] = { 0, 0, 0, 0 };

    // This is actually a macro that modifies the last four operands,
    // hence why they are not passed by address.
    __cpuid(0, eax, vendor_string[0], vendor_string[2], vendor_string[1]);

    // Check extended feature bits for post-AVX2 features.
    if (cpuid_max >= 7) {
        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        __cpuid_count(7, 0, eax, ebx, ecx, edx);

        if (dlp_cpuid_has_features(ebx, FEATURE_MASK_AVX2))
            *features |= FEATURE_AVX2;
        if (dlp_cpuid_has_features(ebx, FEATURE_MASK_AVX512F))
            *features |= FEATURE_AVX512F;
        if (dlp_cpuid_has_features(ebx, FEATURE_MASK_AVX512DQ))
            *features |= FEATURE_AVX512DQ;
        if (dlp_cpuid_has_features(ebx, FEATURE_MASK_AVX512PF))
            *features |= FEATURE_AVX512PF;
        if (dlp_cpuid_has_features(ebx, FEATURE_MASK_AVX512ER))
            *features |= FEATURE_AVX512ER;
        if (dlp_cpuid_has_features(ebx, FEATURE_MASK_AVX512CD))
            *features |= FEATURE_AVX512CD;
        if (dlp_cpuid_has_features(ebx, FEATURE_MASK_AVX512BW))
            *features |= FEATURE_AVX512BW;
        if (dlp_cpuid_has_features(ebx, FEATURE_MASK_AVX512VL))
            *features |= FEATURE_AVX512VL;

        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_AVX512VNNI))
            *features |= FEATURE_AVX512VNNI;
        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_MOVDIRI))
            *features |= FEATURE_MOVDIRI;
        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_MOVDIR64B))
            *features |= FEATURE_MOVDIR64B;

        if (dlp_cpuid_has_features(edx, FEATURE_MASK_AVX512VP2INTERSECT))
            *features |= FEATURE_AVX512VP2INTERSECT;

        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        // This returns extended feature flags in EAX.
        // The availability of AVX512_BF16  can be found using the
        // 5th feature bit of the returned value
        __cpuid_count(7, 1, eax, ebx, ecx, edx);

        if (dlp_cpuid_has_features(eax, FEATURE_MASK_AVXVNNI))
            *features |= FEATURE_AVXVNNI;
        if (dlp_cpuid_has_features(eax, FEATURE_MASK_AVX512BF16))
            *features |= FEATURE_AVX512BF16;
    }

    // Check extended processor info / features bits for AMD-specific features.
    if (cpuid_max_ext >= 0x80000001u) {
        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        __cpuid(0x80000001u, eax, ebx, ecx, edx);

        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_FMA4))
            *features |= FEATURE_FMA4;
    }
    if (cpuid_max_ext >= 0x8000001Au) {
        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        // This returns extended feature flags in EAX.
        __cpuid(0x8000001A, eax, ebx, ecx, edx);

        if (dlp_cpuid_has_features(eax, FEATURE_MASK_DATAPATH_FP128))
            *features |= FEATURE_DATAPATH_FP128;
        if (dlp_cpuid_has_features(eax, FEATURE_MASK_DATAPATH_FP256))
            *features |= FEATURE_DATAPATH_FP256;
        if (dlp_cpuid_has_features(eax, FEATURE_MASK_DATAPATH_FP512))
            *features |= FEATURE_DATAPATH_FP512;
    }

    // Unconditionally check processor info / features bits.
    {
        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        __cpuid(1, eax, ebx, ecx, edx);

        // Check for SSE, AVX, and FMA3 features.
        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_SSE3))
            *features |= FEATURE_SSE3;
        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_SSSE3))
            *features |= FEATURE_SSSE3;
        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_SSE41))
            *features |= FEATURE_SSE41;
        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_SSE42))
            *features |= FEATURE_SSE42;
        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_AVX))
            *features |= FEATURE_AVX;
        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_FMA3))
            *features |= FEATURE_FMA3;

        // Check whether the hardware supports xsave/xrestor/xsetbv/xgetbv AND
        // support for these is enabled by the OS. If so, then we proceed with
        // checking that various register-state saving features are available.
        if (dlp_cpuid_has_features(ecx, FEATURE_MASK_XGETBV)) {
            uint32_t xcr = 0;

            // Call xgetbv to get xcr0 (the extended control register) copied
            // to [edx:eax]. This encodes whether software supports various
            // register state-saving features.
            __asm__ __volatile__(".byte 0x0F, 0x01, 0xD0"
                                 : "=a"(eax), "=d"(edx)
                                 : "c"(xcr)
                                 : "cc");

            // The OS can manage the state of 512-bit zmm (AVX-512) registers
            // only if the xcr[7:5] bits are set. If they are not set, then
            // clear all feature bits related to AVX-512.
            if (!dlp_cpuid_has_features(eax, XGETBV_MASK_XMM | XGETBV_MASK_YMM
                                                 | XGETBV_MASK_ZMM)) {
                *features &=
                    ~(FEATURE_AVX512F | FEATURE_AVX512DQ | FEATURE_AVX512PF
                      | FEATURE_AVX512ER | FEATURE_AVX512CD | FEATURE_AVX512BW
                      | FEATURE_AVX512VL);
            }

            // The OS can manage the state of 256-bit ymm (AVX) registers
            // only if the xcr[2] bit is set. If it is not set, then
            // clear all feature bits related to AVX.
            if (!dlp_cpuid_has_features(eax,
                                        XGETBV_MASK_XMM | XGETBV_MASK_YMM)) {
                *features &=
                    ~(FEATURE_AVX | FEATURE_AVX2 | FEATURE_FMA3 | FEATURE_FMA4);
            }

            // The OS can manage the state of 128-bit xmm (SSE) registers
            // only if the xcr[1] bit is set. If it is not set, then
            // clear all feature bits related to SSE (which means the
            // entire bitfield is clear).
            if (!dlp_cpuid_has_features(eax, XGETBV_MASK_XMM)) {
                *features = 0;
            }
        } else {
            // If the hardware does not support xsave/xrestor/xsetbv/xgetbv,
            // OR these features are not enabled by the OS, then we clear
            // the bitfield, because it means that not even xmm support is
            // present.

            features = 0;
        }
    }

    // Check the vendor string and return a value to indicate Intel or AMD.
    if (strcmp((char*)vendor_string, "AuthenticAMD") == 0) {
        return VENDOR_AMD;
    } else if (strcmp((char*)vendor_string, "GenuineIntel") == 0) {
        return VENDOR_INTEL;
    } else {
        return VENDOR_UNKNOWN;
    }
}

static bool
dlp_cpuid_is_zen5(uint32_t features)
{
    // Check for expected CPU features.
    const uint32_t expected =
        FEATURE_SSE3 | FEATURE_SSSE3 | FEATURE_SSE41 | FEATURE_SSE42
        | FEATURE_AVX | FEATURE_FMA3 | FEATURE_AVX2 | FEATURE_AVX512F
        | FEATURE_AVX512DQ | FEATURE_AVX512CD | FEATURE_AVX512BW
        | FEATURE_AVX512VL | FEATURE_AVX512VNNI | FEATURE_AVX512BF16
        | FEATURE_MOVDIRI | FEATURE_MOVDIR64B | FEATURE_AVX512VP2INTERSECT
        | FEATURE_AVXVNNI;

    if (!dlp_cpuid_has_features(features, expected)) {
        return FALSE;
    }

    return TRUE;
}

bool
dlp_cpuid_is_zen4(uint32_t features)
{
    // Check for expected CPU features.
    const uint32_t expected =
        FEATURE_SSE3 | FEATURE_SSSE3 | FEATURE_SSE41 | FEATURE_SSE42
        | FEATURE_AVX | FEATURE_FMA3 | FEATURE_AVX2 | FEATURE_AVX512F
        | FEATURE_AVX512DQ | FEATURE_AVX512CD | FEATURE_AVX512BW
        | FEATURE_AVX512VL | FEATURE_AVX512VNNI | FEATURE_AVX512BF16;

    if (!dlp_cpuid_has_features(features, expected)) {
        return FALSE;
    }

    return TRUE;
}

bool
dlp_cpuid_is_zen(uint32_t features)
{
    // Check for expected CPU features.
    const uint32_t expected = FEATURE_AVX | FEATURE_FMA3 | FEATURE_AVX2;

    if (!dlp_cpuid_has_features(features, expected)) {
        return FALSE;
    }

    return TRUE;
}

dlp_arch_t
dlp_cpuid_query_id(void)
{
    uint32_t vendor, features;

    vendor = dlp_cpuid_query(&features);

    if (vendor == VENDOR_INTEL || vendor == VENDOR_AMD) {
        // Check different levels of AVX instruction support.
        dlp_cpuid_check_avx2fma3_support(features);
        dlp_cpuid_check_avx512_support(features);
        dlp_cpuid_check_avx512vnni_support(features);
        dlp_cpuid_check_avx512bf16_support(features);

        // Check FP/SIMD execution datapath
        dlp_cpuid_check_datapath(vendor, features);
    }

    if (vendor == VENDOR_INTEL) {
        // Check for each Intel configuration that is enabled, check for that
        // microarchitecture. We check from most recent to most dated.
#ifdef DLP_CONFIG_ZEN4
        // Even if not optimized for Intel processors, this should
        // generally perform better than skx codepath.
        // Currently only enabled for zen4 and amdzen configurations
        if (is_avx512_supported)
            return DLP_ARCH_ZEN4;
#endif
#ifdef DLP_CONFIG_ZEN3
        // Even if not optimized for Intel processors, this should
        // generally perform better than haswell codepath.
        // Currently only enabled for zen3 and amdzen configurations
        if (is_avx2fma3_supported)
            return DLP_ARCH_ZEN3;
#endif
        // If none of the other sub-configurations were detected, return
        // the 'generic' arch_t id value.
        return DLP_ARCH_GENERIC;
    } else if (vendor == VENDOR_AMD) {
        // The ARCH is decided based on the dlp config set during compile
        // time and the ISA supported. The model and family id is NOT used
        // for determining the same.
#ifdef DLP_CONFIG_ZEN5
        if (dlp_cpuid_is_zen5(features))
            return DLP_ARCH_ZEN5;
#endif
#ifdef DLP_CONFIG_ZEN4
        if (dlp_cpuid_is_zen4(features))
            return DLP_ARCH_ZEN4;
#endif
#ifdef DLP_CONFIG_ZEN5
        // Fallback test for future AMD processors
        // Assume zen5 (if available) is preferable to zen4.
        if (is_avx512_supported)
            return DLP_ARCH_ZEN5;
#endif
#ifdef DLP_CONFIG_ZEN4
        // Fallback test for future AMD processors
        // Use zen4 if zen5 is not available.
        if (is_avx512_supported)
            return DLP_ARCH_ZEN4;
#endif
#ifdef DLP_CONFIG_ZEN3
        if (dlp_cpuid_is_zen(features))
            return DLP_ARCH_ZEN3;
#endif
#ifdef DLP_CONFIG_ZEN2
        if (dlp_cpuid_is_zen(features))
            return DLP_ARCH_ZEN2;
#endif
#ifdef DLP_CONFIG_ZEN
        if (dlp_cpuid_is_zen(features))
            return DLP_ARCH_ZEN;
#endif
#ifdef DLP_CONFIG_ZEN3
        // Fallback test for future AMD processors
        // Use zen3 if AVX512 support is not available but AVX2 is.
        if (is_avx2fma3_supported)
            return DLP_ARCH_ZEN3;
#endif
        // If none of the other sub-configurations were detected, return
        // the 'generic' arch_t id value.
        return DLP_ARCH_GENERIC;
    } else if (vendor == VENDOR_UNKNOWN) {
        return DLP_ARCH_GENERIC;
    }

    return DLP_ARCH_GENERIC;
}

dlp_arch_t
dlp_env_get_var_arch_type(const char* env)
{
    dlp_arch_t r_val = DLP_ARCH_ERROR;
    char*      str   = NULL;
    int        i     = 0;
    int        size  = 0;

    // Query the environment variable and store the result in str.
    str = getenv(env);

    // Set the return value based on the string obtained from getenv().
    if (str != NULL) {
        // convert string to lowercase
        size = strlen(str);
        for (i = 0; i <= size; ++i) {
            str[i] = tolower(str[i]);
        }

        // AMD
        if (strcmp(str, "zen5") == 0) {
            r_val = DLP_ARCH_ZEN5;
        } else if (strcmp(str, "zen4") == 0) {
            r_val = DLP_ARCH_ZEN4;
        } else if (strcmp(str, "zen3") == 0) {
            r_val = DLP_ARCH_ZEN3;
        } else if (strcmp(str, "zen2") == 0) {
            r_val = DLP_ARCH_ZEN2;
        } else if ((strcmp(str, "zen") == 0) || (strcmp(str, "zen1") == 0)) {
            r_val = DLP_ARCH_ZEN;
        }
        // Some aliases for mapping AMD and Intel ISA names to a suitable
        // sub-configuration for each x86-64 processor family.
#if defined(DLP_FAMILY_AMDZEN)
        else if (strcmp(str, "avx512") == 0) {
            r_val = DLP_ARCH_ZEN4;
        } else if (strcmp(str, "avx2") == 0) {
            r_val = DLP_ARCH_ZEN3;
        } else if (strcmp(str, "avx") == 0) {
            r_val = DLP_ARCH_GENERIC;
        } else if ((strcmp(str, "sse4_2") == 0) || (strcmp(str, "sse4.2") == 0)
                   || (strcmp(str, "sse4_1") == 0)
                   || (strcmp(str, "sse4.1") == 0)
                   || (strcmp(str, "sse4a") == 0) || (strcmp(str, "sse4") == 0)
                   || (strcmp(str, "ssse3") == 0) || (strcmp(str, "sse3") == 0)
                   || (strcmp(str, "sse2") == 0)) {
            r_val = DLP_ARCH_GENERIC;
        }
#endif
#if defined(DLP_FAMILY_X86_64)
        else if (strcmp(str, "avx512") == 0) {
            r_val = DLP_ARCH_ZEN4;
        } else if (strcmp(str, "avx2") == 0) {
            r_val = DLP_ARCH_ZEN3;
        }
#endif
    }

    return r_val;
}

void
dlp_arch_set_id(void)
{
    // Get actual hardware arch and model ids.
    actual_arch_id = dlp_cpuid_query_id();

    arch_id = dlp_env_get_var_arch_type("AOCL_ENABLE_INSTRUCTIONS");
    if (arch_id != DLP_ARCH_ERROR) {
        aocl_e_i = TRUE;
    } else {
        // Architecture families.
#if defined DLP_FAMILY_INTEL64 || defined DLP_FAMILY_AMDZEN                    \
    || defined DLP_FAMILY_AMD64_LEGACY || defined DLP_FAMILY_X86_64
        arch_id = actual_arch_id;
#endif

        // AMD microarchitectures.
#ifdef DLP_FAMILY_ZEN5
        arch_id = DLP_ARCH_ZEN5;
#endif
#ifdef DLP_FAMILY_ZEN4
        arch_id = DLP_ARCH_ZEN4;
#endif
#ifdef DLP_FAMILY_ZEN3
        arch_id = DLP_ARCH_ZEN3;
#endif
#ifdef DLP_FAMILY_ZEN2
        arch_id = DLP_ARCH_ZEN2;
#endif
#ifdef DLP_FAMILY_ZEN
        arch_id = DLP_ARCH_ZEN;
#endif
    }
}

void
dlp_arch_set_id_once(void)
{
    dlp_pthread_once(&once_id_init, dlp_arch_set_id);
}

void
dlp_arch_check_id(void)
{
    dlp_arch_set_id_once();

    dlp_arch_t orig_arch_id = arch_id;

    if ((orig_arch_id != DLP_ARCH_ERROR) && (aocl_e_i == TRUE)) {
        // If AVX2 test fails here we assume either:
        // 1. Config was either zen, zen2, zen3, zen4, zen5, haswell or skx,
        //    so there is no fallback code path, hence error checking
        //    above will fail.
        // 2. Config was amdzen, intel64 or x86_64, and will have
        //    generic code path.
        if (!dlp_cpuid_is_avx2fma3_supported()) {
            switch (orig_arch_id) {
                case DLP_ARCH_ZEN5:
                case DLP_ARCH_ZEN4:
                case DLP_ARCH_ZEN3:
                case DLP_ARCH_ZEN2:
                case DLP_ARCH_ZEN:
                    arch_id = actual_arch_id;
                    break;
                default:
                    break;
            }
        }
        // If AVX512 test fails here we assume either:
        // 1. Config was either zen5, zen4 or skx, so there is
        //    no fallback code path, hence error checking
        //    above will fail.
        // 2. Config was amdzen, intel64 or x86_64, and will have
        //    appropriate avx2 code path to try.
        if (!dlp_cpuid_is_avx512_supported()) {
            switch (orig_arch_id) {
                case DLP_ARCH_ZEN5:
                case DLP_ARCH_ZEN4:
                    arch_id = actual_arch_id;
                    break;
                default:
                    break;
            }
        }
    }
}

dlp_arch_t
dlp_get_arch(void)
{
    dlp_pthread_once(&once_id_check, dlp_arch_check_id);

    // Simply return the id that was previously cached.
    return arch_id;
}

bool
dlp_aocl_enable_instruction_query(void)
{
    // Return whether the AOCL_ENABLE_INSTRUCTIONS environment variable is set
    // or not.
    return aocl_e_i;
}

static dlp_pthread_once_t once_check_cpuid_query_id = DLP_PTHREAD_ONCE_INIT;

void
dlp_cpuid_query_id_wrapper(void)
{
    dlp_arch_t __attribute__((unused)) id_w;
    id_w = dlp_cpuid_query_id();
}
void
dlp_cpuid_query_id_once(void)
{
    dlp_pthread_once(&once_check_cpuid_query_id, dlp_cpuid_query_id_wrapper);
}

// API to check if AVX2 and FMA3 are supported or not on the current platform.
bool
dlp_cpuid_is_avx2fma3_supported(void)
{
    dlp_cpuid_query_id_once();
    return is_avx2fma3_supported;
}

// API to check if AVX512 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512_supported(void)
{
    dlp_cpuid_query_id_once();
    return is_avx512_supported;
}

// API to check if AVX512_VNNI is supported or not on the current platform.
bool
dlp_cpuid_is_avx512vnni_supported(void)
{
    dlp_cpuid_query_id_once();
    return is_avx512vnni_supported;
}

// API to check if AVX512_bf16 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512bf16_supported(void)
{
    dlp_cpuid_query_id_once();
    return is_avx512bf16_supported;
}

uint32_t
dlp_cpuid_query_fp_datapath(void)
{
    dlp_cpuid_query_id_once();
    return dlp_fp_datapath;
}

#else

dlp_arch_t
dlp_get_arch(void)
{
    return DLP_ARCH_GENERIC;
}

// API to check if AVX2 and FMA3 are supported or not on the current platform.
bool
dlp_cpuid_is_avx2fma3_supported(void)
{
    return FALSE;
}

// API to check if AVX512 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512_supported(void)
{
    return FALSE;
}

// API to check if AVX512_VNNI is supported or not on the current platform.
bool
dlp_cpuid_is_avx512vnni_supported(void)
{
    return FALSE;
}

// API to check if AVX512_bf16 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512bf16_supported(void)
{
    return FALSE;
}

uint32_t
dlp_cpuid_query_fp_datapath(void)
{
    return 0;
}

#endif
