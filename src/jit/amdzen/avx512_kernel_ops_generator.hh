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

#pragma once

#include "kernel_ops_generator_base.hh"

namespace amdzen::avx512gen {

class kernelOpsGeneratorAvx512 : public gen::kernelOpsGeneratorInterface
{
  public:
    kernelOpsGeneratorAvx512(Xbyak::CodeGenerator* jit,
                             int                   MR,
                             int                   NR,
                             bool                  useMask,
                             int                   cRegStartIdx,
                             int                   cRegCount);
    ~kernelOpsGeneratorAvx512()                               = default;
    kernelOpsGeneratorAvx512(const kernelOpsGeneratorAvx512&) = delete;
    kernelOpsGeneratorAvx512& operator=(const kernelOpsGeneratorAvx512&) =
        delete;
    kernelOpsGeneratorAvx512(kernelOpsGeneratorAvx512&&)            = delete;
    kernelOpsGeneratorAvx512& operator=(kernelOpsGeneratorAvx512&&) = delete;

    dlp::jit::jitGeneratorError generateKernelOps(
        std::vector<dlp::kernel_frame::kernelOpsMetaData>& kernelOps,
        const Xbyak::Reg64& postOpsArgWrapperPtrReg) override;
    dlp::jit::jitGeneratorError bias(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError relu(
        [[maybe_unused]] dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError reluScale(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError geluTanh(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError geluErf(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError clip(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError downscale(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError matadd(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError matmul(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError swish(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError tanh(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError sigmoid(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError embedKernelOpsAttributes() override;

    void advancePostOpsPtr() override;

  private:
    int numRegs  = 32;
    int RegSize  = 512;
    int RegBytes = RegSize / 8;

    int MR, NR, useMask;
    int numFullRegsPerRow;
    int numMaskRegsPerRow;
    int numRegsPerRow;

    int cRegStartIdx, cRegCount;
    int scratchLoadRegIdx, scratchBcstRegIdx;

    // registers used for gelu_tanh
    int num_gelu_regs = 9;
    int const1;
    int const2;
    int x;
    int r;
    int r2;
    int z;
    int dn;
    int x_tanh;
    int q;

    // registers for gelu_erf
    int num_erf_regs = 5;
    int x_erf;

    // registers used for swish. Reusing the gelu_tanh registers.
    int num_swish_regs = 9;

    const Xbyak::Reg64 &regkernelOpsList, &regkernelOpsAttr;
    const Xbyak::Reg64 &regTmp1, &regTmp2, &regTmp3, &regTmp4, &regTmp5,
        &regTmp6, &regTmp7;
    const Xbyak::Reg32 &regTmp4Half, &regTmp5Half;

    Xbyak::CodeGenerator* jit_; // Back reference to access registers and state

    dlp::jit::jitGeneratorError setPostOpsContext();

    // Helper implementations for different storage formats
    template<typename T>
    void biasRowMajorZmm();

    template<typename T>
    void biasColMajorZmm();

    template<typename T>
    void reluScaleZmm();

    template<typename T>
    void geluTanhZmm();

    template<typename T>
    void geluErfZmm();

    template<typename T>
    void swishZmm();

    template<typename T>
    void tanhZmm();

    template<typename T>
    void sigmoidZmm();

    template<typename T>
    void clipZmm();

    template<typename T>
    void ScaleFactorScalarZmm();

    template<typename T>
    void ScaleFactorRowMajorZmm();

    template<typename T>
    void ScaleFactorColMajorZmm();

    template<typename T>
    void ZeroPointScalarZmm();

    template<typename T>
    void ZeroPointRowMajorZmm();

    template<typename T>
    void ZeroPointColMajorZmm();

    enum class matOpType
    {
        matOpAdd,
        matOpMul
    };

    enum class matOpScaleType
    {
        scalar,
        rowVector,
        columnVector
    };

    template<typename T>
    void matOpScaleFactorZmm(matOpType opType, matOpScaleType sclType);

    // TODO: Math Utils, move to different class.
    void POLY_EVAL_6_AVX512();
    void EXPF_AVX512();
    void TANHF_AVX512();
    void GELU_TANH_F32_AVX512_DEF(md_t reg);
    void TANHF_AVX512_DEF(md_t reg);
    void POLY_EVAL_HORNER_16_0_AVX512();
    void ERF_AVX512();
    void GELU_ERF_F32_AVX512_DEF(md_t reg);
    void SWISH_F32_AVX512_DEF(md_t reg);
    void SIGMOID_AVX512_DEF(md_t reg);
    void apply_post_ops_in_high_reg_pressure(const md_t num_post_op_regs,
                                             std::function<void(md_t)> op_fn);
    void store_zmms_in_stack(md_t reg_start_idx, md_t num_regs);
    void get_zmms_from_stack(md_t reg_start_idx, md_t num_regs);

    // Table of constants used in gelu_tanh and gelu_erf.
    // TODO: Clean this up and move to a more appropriate place.
    float gelu_consts[7] = { 0.044715, 0.797884, -2, 0.5, -1, 2, 1 };
    float gelu_macros[6] = { 1.4426950408889634, 1.2582912E7, -88.0f, 88.0f,
                             (float)(1.0 / 0.0), -2147483648 };

    float lpgemm_exp[6] = { 1.0000000754895704,   0.6931472254087585,
                            0.2402210737432219,   0.05550297297702539,
                            0.009676036358193323, 0.001341000536524434 };

    float erf_consts[4] = { 0.707107, 1.0, 0.5, 3.553f };

    float lpgemm_erf[16] = { 1.1283793786592402,    2.5468861568875563E-5,
                             0.3756169877289898,    0.004025179163741976,
                             0.12947984300439994,   0.0412525204794885,
                             0.03918550001070417,   0.07104542913277255,
                             0.05717052146749476,   0.025310822854733135,
                             0.0067305713376882076, 0.0010410692067591445,
                             6.921588102382636E-5,  4.092409485758739E-6,
                             1.033131746125426E-6,  5.2927177513236435E-8 };

    const md_t gelu_consts_off = 0;
    const md_t gelu_macros_off = gelu_consts_off + sizeof(gelu_consts);
    const md_t lpgemm_exp_off  = gelu_macros_off + sizeof(gelu_macros);
    const md_t erf_consts_off  = lpgemm_exp_off + sizeof(lpgemm_exp);
    const md_t lpgemm_erf_off  = erf_consts_off + sizeof(erf_consts);

    Xbyak::Address get_constant(md_t table_off, md_t value_off)
    {
        return jit_->ptr[jit_->rip + tables + table_off
                         + (value_off * (md_t)sizeof(float))];
    }
    Xbyak::Label tables;
};

} // namespace amdzen::avx512gen
