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

#ifndef AOCL_ELTWISE_OPS_INTERFACE_H
#define AOCL_ELTWISE_OPS_INTERFACE_H

#include "classic/aocl_bf16_type.h"
#include "classic/aocl_gemm_post_ops.h"
#include "classic/dlp_base_types.h"

/**
 * @brief Performs element-wise operations on a matrix.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] transa Transpose option for matrix A.
 * @param[in] transb Transpose option for matrix B.
 * @param[in] m Number of rows in matrix A and B.
 * @param[in] n Number of columns in matrix A and B.
 * @param[in] a Pointer to matrix A.
 * @param[in] lda Leading dimension of matrix A.
 * @param[out] b Pointer to matrix B.
 * @param[in] ldb Leading dimension of matrix B.
 * @param[in] post_op_unparsed Pointer to post-operation structures.
 */
#define AOCL_UTIL_ELTWISE_OPS(A_type, B_type, LP_SFX)                          \
    DLP_CLASSIC_EXPORT void aocl_gemm_eltwise_ops_##LP_SFX(                    \
        const char order, const char transa, const char transb, const md_t m,  \
        const md_t n, const A_type* a, const md_t lda, B_type* b,              \
        const md_t ldb, aocl_post_op* post_op_unparsed)

AOCL_UTIL_ELTWISE_OPS(bfloat16, float, bf16of32);
AOCL_UTIL_ELTWISE_OPS(bfloat16, bfloat16, bf16obf16);
AOCL_UTIL_ELTWISE_OPS(float, float, f32of32);
AOCL_UTIL_ELTWISE_OPS(float, bfloat16, f32obf16);
AOCL_UTIL_ELTWISE_OPS(float, int32_t, f32os32);
AOCL_UTIL_ELTWISE_OPS(float, int8_t, f32os8);
AOCL_UTIL_ELTWISE_OPS(float, uint8_t, f32ou8);

#endif // AOCL_ELTWISE_OPS_INTERFACE_H
