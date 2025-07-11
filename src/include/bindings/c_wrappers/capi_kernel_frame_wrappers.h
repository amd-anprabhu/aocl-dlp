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

#ifndef CAPI_KERNEL_FRAME_WRAPPERS_H
#define CAPI_KERNEL_FRAME_WRAPPERS_H

#include "classic/dlp_macros.h"

DLP_BEGIN_EXTERN_C

#include <stdint.h>

#include "classic/aocl_gemm_post_ops.h"
#include "classic/dlp_base_types.h"

typedef enum
{
    DLP_KERNEL_INVALID = 0,
    DLP_KERNEL_U8S8S32OS32,
    DLP_KERNEL_U8S8S32OS8,
    DLP_KERNEL_BF16BF16F32OBF16,
    DLP_KERNEL_BF16BF16F32OF32,
    DLP_KERNEL_F32F32F32OF32,
    DLP_KERNEL_DATATYPE_MAX
} kernel_datatype_t;

typedef struct
{
    // Add kernel ops here
    // The aocl post ops will be parsed to a internal representation
    // and then this struct will be passed to the kernel
} kernel_ops_t;

typedef struct
{
    kernel_datatype_t kDtype;
    md_t              mr;
    md_t              nr;
    void*             kernel_base;
    // kernel_ops_t* kernel_ops;
} dlp_kernel_hndl_t;

dlp_kernel_hndl_t
dlp_init_and_get_kernel_hndl(kernel_datatype_t kDtype,
                             md_t              m,
                             md_t              n,
                             md_t              k,
                             void*             alpha,
                             void*             beta,
                             aocl_post_op*     post_ops,
                             md_t              mr_hint,
                             md_t              nr_hint);

void
dlp_execute_kernel(dlp_kernel_hndl_t kernel_hndl,
                   md_t              m,
                   md_t              n,
                   md_t              k,
                   void*             A,
                   md_t              rs_a,
                   md_t              cs_a,
                   md_t              ps_a,
                   void*             B,
                   md_t              rs_b,
                   md_t              cs_b,
                   void*             C,
                   md_t              rs_c,
                   md_t              cs_c,
                   void*             alpha,
                   void*             beta);

DLP_END_EXTERN_C

#endif // CAPI_KERNEL_FRAME_WRAPPERS_H
