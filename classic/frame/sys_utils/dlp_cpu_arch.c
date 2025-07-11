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

#include "bindings/c_wrappers/capi_cpu_features.h"
#include "classic/aocl_lib_interface_apis.h"
#include "sys_utils/dlp_cpu_arch.h"
#include "sys_utils/lpgemm_sys.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386)                  \
    || defined(_M_IX86)

#include "cpuid.h"

static bool               aocl_e_i       = FALSE;
static dlp_arch_t         arch_id        = -1;
static dlp_arch_t         actual_arch_id = -1;
static dlp_pthread_once_t once_id_init   = DLP_PTHREAD_ONCE_INIT;
static dlp_pthread_once_t once_id_check  = DLP_PTHREAD_ONCE_INIT;

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
    }

    return r_val;
}

void
dlp_arch_set_id(void)
{
    // Get actual hardware arch and model ids.
    actual_arch_id = dlp_cpuid_query_arch_id();

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
    id_w = dlp_cpuid_query_arch_id();
}
void
dlp_cpuid_query_id_once(void)
{
    dlp_pthread_once(&once_check_cpuid_query_id, dlp_cpuid_query_id_wrapper);
}

#else

dlp_arch_t
dlp_get_arch(void)
{
    return DLP_ARCH_GENERIC;
}

#endif
