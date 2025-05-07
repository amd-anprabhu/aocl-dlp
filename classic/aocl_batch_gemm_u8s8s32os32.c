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

#include "aocl_gemm_check.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "logging/lpgemm_logger.h"
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "sys_utils/dlp_cpu_arch.h"
#include "threading/lpgemm_thread_decor_openmp.h"

AOCL_BGEMM_MATMUL(uint8_t, int8_t, int32_t, int32_t, u8s8s32os32)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER("u8s8s32os32", order, transa, transb, batch_size,
                              m, n, k, alpha, lda, mem_format_a, ldb,
                              mem_format_b, beta, ldc, post_op_unparsed);

    md_t rs_a[batch_size];
    md_t cs_a[batch_size];

    md_t rs_b[batch_size];
    md_t cs_b[batch_size];

    md_t rs_c[batch_size];
    md_t cs_c[batch_size];

    AOCL_MEMORY_TAG mtag_a[batch_size];
    AOCL_MEMORY_TAG mtag_b[batch_size];

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[batch_size][AOCL_MAX_POST_OPS];

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32 gemm.",
                      __FILE__, __LINE__);
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;

    // check for validity of params.
    int err_no = 0;

    for (md_t bs_i = 0; bs_i < batch_size; bs_i++) {
        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK(
            "batch_u8s8s32os32", order[bs_i], transa[bs_i], transb[bs_i], bs_i,
            m[bs_i], n[bs_i], k[bs_i], a[bs_i], lda[bs_i], mem_format_a[bs_i],
            b[bs_i], ldb[bs_i], mem_format_b[bs_i], c[bs_i], ldc[bs_i], err_no);

        if (err_no != 0) {
            goto err_hndl;
        }

        /* Map BLAS chars to their corresponding DLP enumerated type value. */
        dlp_param_map_netlib_to_dlp_trans(transa[bs_i], &dlp_transa);
        dlp_param_map_netlib_to_dlp_trans(transb[bs_i], &dlp_transb);

        bool is_column_major = ((order[bs_i] == 'c') || (order[bs_i] == 'C'));

        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            goto err_hndl;
        } else // row-major
        {
            rs_a[bs_i] = lda[bs_i];
            cs_a[bs_i] = 1;

            if (dlp_is_trans(dlp_transa)) {
                rs_a[bs_i] = 1;
                cs_a[bs_i] = lda[bs_i];
            }

            rs_b[bs_i] = ldb[bs_i];
            cs_b[bs_i] = 1;

            if (dlp_is_trans(dlp_transb)) {
                rs_b[bs_i] = 1;
                cs_b[bs_i] = ldb[bs_i];
            }

            dlp_param_map_char_to_lpmtag(mem_format_a[bs_i], &(mtag_a[bs_i]));
            dlp_param_map_char_to_lpmtag(mem_format_b[bs_i], &(mtag_b[bs_i]));

            // Reorder is not supported for A matrix
            if (mtag_a[bs_i] == REORDERED) {
                dlp_print_msg(" Reordering of A matrix is not supported in row "
                              "major case.",
                              __FILE__, __LINE__);
                goto err_hndl;
            }
            // From 5-loop function point of view,
            // A matrix when in column major storage needs to be packed to
            // row-major storage as kernel expects A matrix to be in row-major
            // format.
            if (dlp_is_trans(dlp_transa)) {
                mtag_a[bs_i] = PACK;
            }
        }

        rs_c[bs_i] = ldc[bs_i];
        cs_c[bs_i] = 1;

        // From 5-loop function point of view
        // B matrix needs to be packed in a certain format in order to be loaded
        // and used in bf16 instrution. As such the mtag_b always needs to be
        // either packed or reordered. B matrix as it is (unpacked) cannot be
        // used, and the mtag_b is set to packed to enable runtime packing.
        if (mtag_b[bs_i] == UNPACKED) {
            mtag_b[bs_i] = PACK;
        }

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
            post_op_unparsed[bs_i], post_op_list[bs_i], (void*)c[bs_i],
            (void*)((order + bs_i)), m[bs_i], n[bs_i]);

        if (err != DLP_CLSC_SUCCESS)
            goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);

#ifdef DLP_ENABLE_OPENMP
    batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c,
        rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list, S32);

#else
    batch_lpgemm_u8s8s32o32_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c,
        rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list, S32);
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}

AOCL_BGEMM_MATMUL(uint8_t, int8_t, int8_t, int32_t, u8s8s32os8)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER("u8s8s32os8", order, transa, transb, batch_size,
                              m, n, k, alpha, lda, mem_format_a, ldb,
                              mem_format_b, beta, ldc, post_op_unparsed);

    md_t rs_a[batch_size];
    md_t cs_a[batch_size];

    md_t rs_b[batch_size];
    md_t cs_b[batch_size];

    md_t rs_c[batch_size];
    md_t cs_c[batch_size];

    AOCL_MEMORY_TAG mtag_a[batch_size];
    AOCL_MEMORY_TAG mtag_b[batch_size];

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[batch_size][AOCL_MAX_POST_OPS];

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32 gemm.",
                      __FILE__, __LINE__);
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;

    // check for validity of params.
    int err_no = 0;

    for (md_t bs_i = 0; bs_i < batch_size; bs_i++) {
        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK(
            "batch_u8s8s32os8", order[bs_i], transa[bs_i], transb[bs_i], bs_i,
            m[bs_i], n[bs_i], k[bs_i], a[bs_i], lda[bs_i], mem_format_a[bs_i],
            b[bs_i], ldb[bs_i], mem_format_b[bs_i], c[bs_i], ldc[bs_i], err_no);

        if (err_no != 0) {
            goto err_hndl;
        }

        /* Map BLAS chars to their corresponding DLP enumerated type value. */
        dlp_param_map_netlib_to_dlp_trans(transa[bs_i], &dlp_transa);
        dlp_param_map_netlib_to_dlp_trans(transb[bs_i], &dlp_transb);

        bool is_column_major = ((order[bs_i] == 'c') || (order[bs_i] == 'C'));

        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            goto err_hndl;
        } else // row-major
        {
            rs_a[bs_i] = lda[bs_i];
            cs_a[bs_i] = 1;

            if (dlp_is_trans(dlp_transa)) {
                rs_a[bs_i] = 1;
                cs_a[bs_i] = lda[bs_i];
            }

            rs_b[bs_i] = ldb[bs_i];
            cs_b[bs_i] = 1;

            if (dlp_is_trans(dlp_transb)) {
                rs_b[bs_i] = 1;
                cs_b[bs_i] = ldb[bs_i];
            }

            dlp_param_map_char_to_lpmtag(mem_format_a[bs_i], &(mtag_a[bs_i]));
            dlp_param_map_char_to_lpmtag(mem_format_b[bs_i], &(mtag_b[bs_i]));

            // Reorder is not supported for A matrix
            if (mtag_a[bs_i] == REORDERED) {
                dlp_print_msg(" Reordering of A matrix is not supported in row "
                              "major case.",
                              __FILE__, __LINE__);
                goto err_hndl;
            }
            // From 5-loop function point of view,
            // A matrix when in column major storage needs to be packed to
            // row-major storage as kernel expects A matrix to be in row-major
            // format.
            if (dlp_is_trans(dlp_transa)) {
                mtag_a[bs_i] = PACK;
            }
        }

        rs_c[bs_i] = ldc[bs_i];
        cs_c[bs_i] = 1;

        // From 5-loop function point of view
        // B matrix needs to be packed in a certain format in order to be loaded
        // and used in bf16 instrution. As such the mtag_b always needs to be
        // either packed or reordered. B matrix as it is (unpacked) cannot be
        // used, and the mtag_b is set to packed to enable runtime packing.
        if (mtag_b[bs_i] == UNPACKED) {
            mtag_b[bs_i] = PACK;
        }

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
            post_op_unparsed[bs_i], post_op_list[bs_i], (void*)c[bs_i],
            (void*)((order + bs_i)), m[bs_i], n[bs_i]);

        if (err != DLP_CLSC_SUCCESS)
            goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);

#ifdef DLP_ENABLE_OPENMP
    batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b,
        (int32_t**)c, rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list,
        S8);

#else
    batch_lpgemm_u8s8s32o32_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b,
        (int32_t**)c, rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list,
        S8);
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}

AOCL_BGEMM_MATMUL(uint8_t, int8_t, float, int32_t, u8s8s32of32)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER("u8s8s32of32", order, transa, transb, batch_size,
                              m, n, k, alpha, lda, mem_format_a, ldb,
                              mem_format_b, beta, ldc, post_op_unparsed);

    md_t rs_a[batch_size];
    md_t cs_a[batch_size];

    md_t rs_b[batch_size];
    md_t cs_b[batch_size];

    md_t rs_c[batch_size];
    md_t cs_c[batch_size];

    AOCL_MEMORY_TAG mtag_a[batch_size];
    AOCL_MEMORY_TAG mtag_b[batch_size];

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[batch_size][AOCL_MAX_POST_OPS];

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32of32 gemm.",
                      __FILE__, __LINE__);
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;

    // check for validity of params.
    int err_no = 0;

    for (md_t bs_i = 0; bs_i < batch_size; bs_i++) {
        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK(
            "batch_u8s8s32of32", order[bs_i], transa[bs_i], transb[bs_i], bs_i,
            m[bs_i], n[bs_i], k[bs_i], a[bs_i], lda[bs_i], mem_format_a[bs_i],
            b[bs_i], ldb[bs_i], mem_format_b[bs_i], c[bs_i], ldc[bs_i], err_no);

        if (err_no != 0) {
            goto err_hndl;
        }

        /* Map BLAS chars to their corresponding DLP enumerated type value. */
        dlp_param_map_netlib_to_dlp_trans(transa[bs_i], &dlp_transa);
        dlp_param_map_netlib_to_dlp_trans(transb[bs_i], &dlp_transb);

        bool is_column_major = ((order[bs_i] == 'c') || (order[bs_i] == 'C'));

        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            goto err_hndl;
        } else // row-major
        {
            rs_a[bs_i] = lda[bs_i];
            cs_a[bs_i] = 1;

            if (dlp_is_trans(dlp_transa)) {
                rs_a[bs_i] = 1;
                cs_a[bs_i] = lda[bs_i];
            }

            rs_b[bs_i] = ldb[bs_i];
            cs_b[bs_i] = 1;

            if (dlp_is_trans(dlp_transb)) {
                rs_b[bs_i] = 1;
                cs_b[bs_i] = ldb[bs_i];
            }

            dlp_param_map_char_to_lpmtag(mem_format_a[bs_i], &(mtag_a[bs_i]));
            dlp_param_map_char_to_lpmtag(mem_format_b[bs_i], &(mtag_b[bs_i]));

            // Reorder is not supported for A matrix
            if (mtag_a[bs_i] == REORDERED) {
                dlp_print_msg(" Reordering of A matrix is not supported in row "
                              "major case.",
                              __FILE__, __LINE__);
                goto err_hndl;
            }
            // From 5-loop function point of view,
            // A matrix when in column major storage needs to be packed to
            // row-major storage as kernel expects A matrix to be in row-major
            // format.
            if (dlp_is_trans(dlp_transa)) {
                mtag_a[bs_i] = PACK;
            }
        }

        rs_c[bs_i] = ldc[bs_i];
        cs_c[bs_i] = 1;

        // From 5-loop function point of view
        // B matrix needs to be packed in a certain format in order to be loaded
        // and used in bf16 instrution. As such the mtag_b always needs to be
        // either packed or reordered. B matrix as it is (unpacked) cannot be
        // used, and the mtag_b is set to packed to enable runtime packing.
        if (mtag_b[bs_i] == UNPACKED) {
            mtag_b[bs_i] = PACK;
        }

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
            post_op_unparsed[bs_i], post_op_list[bs_i], (void*)c[bs_i],
            (void*)((order + bs_i)), m[bs_i], n[bs_i]);

        if (err != DLP_CLSC_SUCCESS)
            goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);

#ifdef DLP_ENABLE_OPENMP
    batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b,
        (int32_t**)c, rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list,
        F32);

#else
    batch_lpgemm_u8s8s32o32_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b,
        (int32_t**)c, rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list,
        F32);
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}

AOCL_BGEMM_MATMUL(uint8_t, int8_t, bfloat16, int32_t, u8s8s32obf16)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER("u8s8s32obf16", order, transa, transb, batch_size,
                              m, n, k, alpha, lda, mem_format_a, ldb,
                              mem_format_b, beta, ldc, post_op_unparsed);

    md_t rs_a[batch_size];
    md_t cs_a[batch_size];

    md_t rs_b[batch_size];
    md_t cs_b[batch_size];

    md_t rs_c[batch_size];
    md_t cs_c[batch_size];

    AOCL_MEMORY_TAG mtag_a[batch_size];
    AOCL_MEMORY_TAG mtag_b[batch_size];

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[batch_size][AOCL_MAX_POST_OPS];

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32obf16 gemm.",
                      __FILE__, __LINE__);
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;

    // check for validity of params.
    int err_no = 0;

    for (md_t bs_i = 0; bs_i < batch_size; bs_i++) {
        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK(
            "batch_u8s8s32obf16", order[bs_i], transa[bs_i], transb[bs_i], bs_i,
            m[bs_i], n[bs_i], k[bs_i], a[bs_i], lda[bs_i], mem_format_a[bs_i],
            b[bs_i], ldb[bs_i], mem_format_b[bs_i], c[bs_i], ldc[bs_i], err_no);

        if (err_no != 0) {
            goto err_hndl;
        }

        /* Map BLAS chars to their corresponding DLP enumerated type value. */
        dlp_param_map_netlib_to_dlp_trans(transa[bs_i], &dlp_transa);
        dlp_param_map_netlib_to_dlp_trans(transb[bs_i], &dlp_transb);

        bool is_column_major = ((order[bs_i] == 'c') || (order[bs_i] == 'C'));

        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            goto err_hndl;
        } else // row-major
        {
            rs_a[bs_i] = lda[bs_i];
            cs_a[bs_i] = 1;

            if (dlp_is_trans(dlp_transa)) {
                rs_a[bs_i] = 1;
                cs_a[bs_i] = lda[bs_i];
            }

            rs_b[bs_i] = ldb[bs_i];
            cs_b[bs_i] = 1;

            if (dlp_is_trans(dlp_transb)) {
                rs_b[bs_i] = 1;
                cs_b[bs_i] = ldb[bs_i];
            }

            dlp_param_map_char_to_lpmtag(mem_format_a[bs_i], &(mtag_a[bs_i]));
            dlp_param_map_char_to_lpmtag(mem_format_b[bs_i], &(mtag_b[bs_i]));

            // Reorder is not supported for A matrix
            if (mtag_a[bs_i] == REORDERED) {
                dlp_print_msg(" Reordering of A matrix is not supported in row "
                              "major case.",
                              __FILE__, __LINE__);
                goto err_hndl;
            }
            // From 5-loop function point of view,
            // A matrix when in column major storage needs to be packed to
            // row-major storage as kernel expects A matrix to be in row-major
            // format.
            if (dlp_is_trans(dlp_transa)) {
                mtag_a[bs_i] = PACK;
            }
        }

        rs_c[bs_i] = ldc[bs_i];
        cs_c[bs_i] = 1;

        // From 5-loop function point of view
        // B matrix needs to be packed in a certain format in order to be loaded
        // and used in bf16 instrution. As such the mtag_b always needs to be
        // either packed or reordered. B matrix as it is (unpacked) cannot be
        // used, and the mtag_b is set to packed to enable runtime packing.
        if (mtag_b[bs_i] == UNPACKED) {
            mtag_b[bs_i] = PACK;
        }

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
            post_op_unparsed[bs_i], post_op_list[bs_i], (void*)c[bs_i],
            (void*)((order + bs_i)), m[bs_i], n[bs_i]);

        if (err != DLP_CLSC_SUCCESS)
            goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);

#ifdef DLP_ENABLE_OPENMP
    batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b,
        (int32_t**)c, rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list,
        BF16);

#else
    batch_lpgemm_u8s8s32o32_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b,
        (int32_t**)c, rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list,
        BF16);
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}

AOCL_BGEMM_MATMUL(uint8_t, int8_t, uint8_t, int32_t, u8s8s32ou8)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER("u8s8s32ou8", order, transa, transb, batch_size,
                              m, n, k, alpha, lda, mem_format_a, ldb,
                              mem_format_b, beta, ldc, post_op_unparsed);

    md_t rs_a[batch_size];
    md_t cs_a[batch_size];

    md_t rs_b[batch_size];
    md_t cs_b[batch_size];

    md_t rs_c[batch_size];
    md_t cs_c[batch_size];

    AOCL_MEMORY_TAG mtag_a[batch_size];
    AOCL_MEMORY_TAG mtag_b[batch_size];

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[batch_size][AOCL_MAX_POST_OPS];

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32ou8 gemm.",
                      __FILE__, __LINE__);
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;

    // check for validity of params.
    int err_no = 0;

    for (md_t bs_i = 0; bs_i < batch_size; bs_i++) {
        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK(
            "batch_u8s8s32ou8", order[bs_i], transa[bs_i], transb[bs_i], bs_i,
            m[bs_i], n[bs_i], k[bs_i], a[bs_i], lda[bs_i], mem_format_a[bs_i],
            b[bs_i], ldb[bs_i], mem_format_b[bs_i], c[bs_i], ldc[bs_i], err_no);

        if (err_no != 0) {
            goto err_hndl;
        }

        /* Map BLAS chars to their corresponding DLP enumerated type value. */
        dlp_param_map_netlib_to_dlp_trans(transa[bs_i], &dlp_transa);
        dlp_param_map_netlib_to_dlp_trans(transb[bs_i], &dlp_transb);

        bool is_column_major = ((order[bs_i] == 'c') || (order[bs_i] == 'C'));

        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            goto err_hndl;
        } else // row-major
        {
            rs_a[bs_i] = lda[bs_i];
            cs_a[bs_i] = 1;

            if (dlp_is_trans(dlp_transa)) {
                rs_a[bs_i] = 1;
                cs_a[bs_i] = lda[bs_i];
            }

            rs_b[bs_i] = ldb[bs_i];
            cs_b[bs_i] = 1;

            if (dlp_is_trans(dlp_transb)) {
                rs_b[bs_i] = 1;
                cs_b[bs_i] = ldb[bs_i];
            }

            dlp_param_map_char_to_lpmtag(mem_format_a[bs_i], &(mtag_a[bs_i]));
            dlp_param_map_char_to_lpmtag(mem_format_b[bs_i], &(mtag_b[bs_i]));

            // Reorder is not supported for A matrix
            if (mtag_a[bs_i] == REORDERED) {
                dlp_print_msg(" Reordering of A matrix is not supported in row "
                              "major case.",
                              __FILE__, __LINE__);
                goto err_hndl;
            }
            // From 5-loop function point of view,
            // A matrix when in column major storage needs to be packed to
            // row-major storage as kernel expects A matrix to be in row-major
            // format.
            if (dlp_is_trans(dlp_transa)) {
                mtag_a[bs_i] = PACK;
            }
        }

        rs_c[bs_i] = ldc[bs_i];
        cs_c[bs_i] = 1;

        // From 5-loop function point of view
        // B matrix needs to be packed in a certain format in order to be loaded
        // and used in bf16 instrution. As such the mtag_b always needs to be
        // either packed or reordered. B matrix as it is (unpacked) cannot be
        // used, and the mtag_b is set to packed to enable runtime packing.
        if (mtag_b[bs_i] == UNPACKED) {
            mtag_b[bs_i] = PACK;
        }

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
            post_op_unparsed[bs_i], post_op_list[bs_i], (void*)c[bs_i],
            (void*)((order + bs_i)), m[bs_i], n[bs_i]);

        if (err != DLP_CLSC_SUCCESS)
            goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);

#ifdef DLP_ENABLE_OPENMP
    batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b,
        (int32_t**)c, rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list,
        U8);

#else
    batch_lpgemm_u8s8s32o32_thread_decorator(
        batch_size, m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b,
        (int32_t**)c, rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, post_op_list,
        U8);
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}
