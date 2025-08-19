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

#include <functional>
#include <memory>

#include "aocl_dlp_config.h"

#if DLP_OS_WINDOWS
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "avx512_gemv.hh"
#include "jit_register/jit_register.hh"

namespace amdzen::avx512gen {

jitAVX512GEMVN1::jitAVX512GEMVN1(void* buffer, size_t size)
    : Xbyak::CodeGenerator(size, buffer) // Call base class constructor
{
}

void
jitAVX512GEMVN1::initializeStackFrame(Xbyak::util::StackFrame& frame)
{
    stackPtr   = frame.p[0];
    regAptr    = frame.t[0];
    regTmpAptr = frame.t[1];
    regXptr    = frame.t[2];
    regYptr    = frame.t[3];
    regTmpYptr = frame.t[4];
    regRsA     = frame.t[5];
    regCsA     = frame.t[6];
    regRsC     = frame.t[7];
    regMIter   = frame.t[8];
    regKIter   = frame.t[9];
    regTmp1    = frame.t[10];
    regTmp2    = frame.t[11];
    regTmp3    = frame.t[12];
}

void
jitAVX512GEMVN1::regInitZmm(int baseIdx, int numRegs)
{
    // Zero out accumulation registers
    vxorps(Xbyak::Zmm(baseIdx), Xbyak::Zmm(baseIdx), Xbyak::Zmm(baseIdx));
    for (int i = 0; i < numRegs; i++) {
        vmovaps(Xbyak::Zmm(baseIdx + i), Xbyak::Zmm(baseIdx));
    }
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::allocateRegisters<float>()
{
    // Check if MR is valid
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Allocate registers according to the rules:
    // 1. Accumulation registers : MR registers for partial dot products
    accumReg     = MR;
    accumBaseIdx = numRegs - accumReg; // Start from the end

    yReg     = (MR + 15) / (RegBytes / sizeof(float));
    yBaseIdx = numRegs - yReg; // Start from the end

    // NOTE : Before loading from y, we would be using MR registers from the end
    //        for accumulating alpha*A*B. This would then be reduced to MR/16
    //        registers, starting from accumBaseIdx. We would still have
    //        15*MR/16 registers left, which we would use for storing the
    //        result, indexed from yBaseIdx(which would be the last MR/16
    //        registers).

    // Ex : If MR is 16
    //      accumReg = 16
    //      accumBaseIdx = 32 - 16 = 16
    //      yReg = 16 / (64 / 4) = 1
    //      yBaseIdx = 31
    //      tmpReg = 4
    //      tmpBaseIdx = 0
    //      xReg = 31 - 16 - 4 = 11
    //      xBaseIdx = 11

    // Temporary registers (tmpReg): Use remaining registers for reduction
    tmpReg     = 4;
    tmpBaseIdx = 0; // To make sure we index YMM greater than 16

    // X registers (xReg): Use remaining registers for vector x
    // We need to only consider accumReg, tmpReg and xReg for total register
    // count.
    xReg     = numRegs - accumReg - tmpReg;
    xBaseIdx = tmpReg;

    // Check if we have enough registers
    if (xReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
void
jitAVX512GEMVN1::initializeParameters<float, float, float>(
    const utils::gemvGeneratorParams& params)
{
    // TODO : Reimplement base on the the latest runtime params struct
    // Set dimensions from params
    MR               = params.MR; // Number of rows to process
    M_LEFT           = params.M_LEFT;
    cMatFormat       = params.cMatFormat;       // Storage format of C matrix
    alphaScalingType = params.alphaScalingType; // Type of alpha scaling
    betaScalingType  = params.betaScalingType;  // Type of beta scaling

    // Load pointers and strides from the stack
    // mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, a)]);
    // mov(regXptr, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, x)]);
    // mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, y)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, csA)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, rsC)]);

    // Scale strides by data type size
    lea(regRsA, ptr[regRsA * sizeof(float)]);
    lea(regCsA, ptr[regCsA * sizeof(float)]);
    lea(regRsC, ptr[regRsC * sizeof(float)]);
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::loadAValues<float>(int aRegIdx, int rowIdx, bool isFringe)
{
    // Calculate row offset first
    mov(regTmp2, rowIdx);
    imul(regTmp2, regRsA); // regTmp1 = rowIdx * regRsA

    // Load 16 or lesser elements, based on whether it is a fringe case or not.
    if (isFringe) {
        vmovups(Xbyak::Zmm(aBaseIdx + aRegIdx) | k3, ptr[regTmpAptr + regTmp2]);
    } else {
        vmovups(Xbyak::Zmm(aBaseIdx + aRegIdx), ptr[regTmpAptr + regTmp2]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::loadXValues<float>(bool isFringe)
{
    // Load 16 or lesser elements, based on whether it is a fringe case or not.
    if (isFringe) {
        vmovups(Xbyak::Zmm(xBaseIdx) | k3, ptr[regXptr]);
    } else {
        vmovups(Xbyak::Zmm(xBaseIdx), ptr[regXptr]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::loadYValues<float>(int yIdx)
{
    // Load values from Y
    vmovups(Xbyak::Zmm(yBaseIdx + yIdx), ptr[regTmp2]);

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::computeFMA<float, float, float>(int aRegIdx, int accumRegIdx)
{
    vfmadd231ps(Xbyak::Zmm(accumBaseIdx + accumRegIdx),
                Xbyak::Zmm(aBaseIdx + aRegIdx), Xbyak::Zmm(xBaseIdx));

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::computeLoadFMA<float, float, float>(int rowIdx, bool isFringe)
{
    if (isFringe) {

        switch (rowIdx % 8) {
            case 0:
            case 1:
            case 2:
            case 4:
                // Direct scaling for 0,1,2,4
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpAptr + regRsA * (rowIdx % 8)]);
                break;
            case 3:
                // Use pre-calculated 3*rsA from regTmp2
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp1]);
                break;
            case 5:
                // Use pre-calculated 5*rsA from regTmp1
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp2]);
                break;
            case 6:
                // Use 2*3*rsA (scale regTmp2 by 2)
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpAptr + regTmp1 * 2]);
                break;
            case 7:
                // Use pre-calculated 7*rsA
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp3]);
                break;
        }
    } else {
        // Same cases without masking
        switch (rowIdx % 8) {
            case 0:
            case 1:
            case 2:
            case 4:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpAptr + regRsA * (rowIdx % 8)]);
                break;
            case 3:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp1]);
                break;
            case 5:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp2]);
                break;
            case 6:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpAptr + regTmp1 * 2]);
                break;
            case 7:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp3]);
                break;
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::reduce4ZMMtoXMM<float>(int startIdx, int tmpIdx, int blockSize)
{
    // Function only handles blocks of 4 or fewer ZMMs
    if (blockSize > 4) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Zero out the temporary registers we'll need
    for (int i = 0; i < 4; i++) {
        vxorps(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(tmpIdx + i),
               Xbyak::Ymm(tmpIdx + i));
    }

    // Extract upper 256-bits and add to lower 256-bits for valid inputs
    for (int i = 0; i < blockSize; i++) {
        // Extract upper 256-bits to temp YMM
        vextractf32x8(Xbyak::Ymm(tmpIdx + i), Xbyak::Zmm(startIdx + i), 1);
        // Add to lower 256-bits of input ZMM, storing in original ZMM's YMM
        // part
        vaddps(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(tmpIdx + i),
               Xbyak::Ymm(startIdx + i));
    }

    // First round of horizontal adds
    vhaddps(Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx),
            Xbyak::Ymm(tmpIdx + 1)); // First pair (with zero if blockSize=1)

    // Second round of horizontal adds
    vhaddps(Xbyak::Ymm(tmpIdx + 2), Xbyak::Ymm(tmpIdx + 2),
            Xbyak::Ymm(tmpIdx + 3));

    // Third round of horizontal adds
    vhaddps(Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx + 2));

    // Final reduction from YMM to XMM
    vextractf128(Xbyak::Xmm(tmpIdx + 1), Xbyak::Ymm(tmpIdx),
                 1); // Extract upper 128-bits
    vaddps(Xbyak::Xmm(tmpIdx), Xbyak::Xmm(tmpIdx + 1),
           Xbyak::Xmm(tmpIdx)); // Add to lower 128-bits

    // Result is now in the XMM portion of startIdx
    // - For blockSize=1: Only first float is valid
    // - For blockSize=2: First two floats are valid
    // - For blockSize=3: First three floats are valid
    // - For blockSize=4: All four floats are valid
    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::reduceAccumulation<float>(int mSize)
{
    // Process mSize registers in blocks of 16 (ZMM width for f32)

    for (int i = 0; i < mSize; i += 16) {
        // Number of registers to process in this ZMM block
        int blockSize = (mSize - i) < 16 ? (mSize - i) : 16;

        // Process this ZMM block in groups of 4 registers
        for (int j = 0; j < blockSize; j += 4) {
            int subBlockSize = (blockSize - j) < 4 ? (blockSize - j) : 4;

            // Reduce 4 (or fewer) ZMMs to one XMM
            RETURN_IF_ERROR((reduce4ZMMtoXMM<float>(accumBaseIdx + i + j,
                                                    tmpBaseIdx, subBlockSize)));

            // Insert the resulting XMM
            // into the appropriate
            // position in destination
            // ZMM
            vinsertf32x4(Xbyak::Zmm(accumBaseIdx + i / 16),
                         Xbyak::Zmm(accumBaseIdx + i / 16),
                         Xbyak::Xmm(tmpBaseIdx), j / 4);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::scaleAccumulationWithAlpha<float>(int mSize)
{
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, alpha)]);
    vbroadcastss(Xbyak::Zmm(tmpBaseIdx), ptr[regKIter]);
    for (int i = 0; i < (mSize + 15) / 16; i += 1) {
        vmulps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(accumBaseIdx + i),
               Xbyak::Zmm(tmpBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::scaleYWithBetaColStored<float>(int  mSize,
                                                bool betaOne,
                                                bool maskType)
{
    if (!betaOne) {
        mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, beta)]);
        vbroadcastss(Xbyak::Zmm(xBaseIdx), ptr[regKIter]);
    }
    int mLeft = mSize % 16;
    for (int i = 0; i < mSize / 16; i += 1) {
        if (betaOne) {
            vaddps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(accumBaseIdx + i),
                   ptr[regTmpYptr]);
        } else {
            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(xBaseIdx),
                        ptr[regTmpYptr]);
        }
        lea(regTmpYptr, ptr[regTmpYptr + 16 * sizeof(float)]);
    }
    if (mLeft) {
        if (!maskType) {
            kmovw(k3,
                  ptr[stackPtr + offsetof(dlp::kernels::gemvParams, mr_mask)]);
        } else {
            kmovw(k3,
                  ptr[stackPtr + offsetof(dlp::kernels::gemvParams, m_mask)]);
        }
        if (betaOne) {
            vaddps(Xbyak::Zmm(accumBaseIdx + (mSize / 16)) | k3,
                   Xbyak::Zmm(accumBaseIdx + (mSize / 16)), ptr[regTmpYptr]);
        } else {
            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + (mSize / 16)) | k3,
                        Xbyak::Zmm(xBaseIdx), ptr[regTmpYptr]);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::scaleYWithBetaRowStored<float>(int mSize, bool betaOne)
{
    if (!betaOne) {
        mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, beta)]);
        vbroadcastss(Xbyak::Zmm(xBaseIdx), ptr[regKIter]);
    }
    // Store offsets for Y, using it's row-stride
    lea(regTmp3, ptr[regRsC + 2 * regRsC]); // regTmp3 = rsC + 2*rsC
    for (int i = 0; i < (mSize + 15) / 16; i += 1) {
        int blockSize  = (mSize - i * 16) < 16 ? (mSize - i * 16) : 16;
        int num_blocks = blockSize / 4;
        int rem_block  = blockSize % 4;
        regInitZmm(tmpBaseIdx, tmpReg);
        for (int j = 0; j < num_blocks; j += 1) {
            vbroadcastss(Xbyak::Zmm(tmpBaseIdx), ptr[regTmpYptr]);
            vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 1), ptr[regTmpYptr + regRsC]);
            vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 2),
                         ptr[regTmpYptr + 2 * regRsC]);
            vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 3), ptr[regTmpYptr + regTmp3]);
            vunpcklps(Xbyak::Zmm(tmpBaseIdx), Xbyak::Zmm(tmpBaseIdx),
                      Xbyak::Zmm(tmpBaseIdx + 1));
            vunpcklps(Xbyak::Zmm(tmpBaseIdx + 2), Xbyak::Zmm(tmpBaseIdx + 2),
                      Xbyak::Zmm(tmpBaseIdx + 3));
            vshufps(Xbyak::Zmm(tmpBaseIdx), Xbyak::Zmm(tmpBaseIdx),
                    Xbyak::Zmm(tmpBaseIdx + 2), 0x44);
            vinsertf32x4(Xbyak::Zmm(yBaseIdx + i), Xbyak::Zmm(yBaseIdx + i),
                         Xbyak::Xmm(tmpBaseIdx), j);
            lea(regTmpYptr, ptr[regTmpYptr + regRsC * 4]);
        }
        if (rem_block) {
            switch (rem_block) {
                case 3:
                    vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 2),
                                 ptr[regTmpYptr + regRsC * 2]);
                case 2:
                    vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 1),
                                 ptr[regTmpYptr + regRsC]);
                case 1:
                    vbroadcastss(Xbyak::Zmm(tmpBaseIdx), ptr[regTmpYptr]);
                case 0:
                    break;
            }
            vunpcklps(Xbyak::Zmm(tmpBaseIdx), Xbyak::Zmm(tmpBaseIdx),
                      Xbyak::Zmm(tmpBaseIdx + 1));
            vunpcklps(Xbyak::Zmm(tmpBaseIdx + 2), Xbyak::Zmm(tmpBaseIdx + 2),
                      Xbyak::Zmm(tmpBaseIdx + 3));
            vshufps(Xbyak::Zmm(tmpBaseIdx), Xbyak::Zmm(tmpBaseIdx),
                    Xbyak::Zmm(tmpBaseIdx + 2), 0x44);
            vinsertf32x4(Xbyak::Zmm(yBaseIdx + i), Xbyak::Zmm(yBaseIdx + i),
                         Xbyak::Xmm(tmpBaseIdx), num_blocks);
        }

        if (betaOne) {
            vaddps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(accumBaseIdx + i),
                   Xbyak::Zmm(yBaseIdx + i));
        } else {
            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(xBaseIdx),
                        Xbyak::Zmm(yBaseIdx + i));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::scaleYWithBeta<float>(int mSize, bool maskType)
{
    bool is_beta_one = (betaScalingType == dlp::kernel_frame::scalingType::one);
    if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
        mov(regTmpYptr, regYptr);
        if (cMatFormat == dlp::kernel_frame::storageFormat::colMajor) {
            RETURN_IF_ERROR(
                (scaleYWithBetaColStored<float>(mSize, is_beta_one, maskType)));
        } else {
            RETURN_IF_ERROR(
                (scaleYWithBetaRowStored<float>(mSize, is_beta_one)));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::storeYValuesColStored<float>(int mSize, bool maskType)
{
    int mLeft = mSize % 16;
    for (int i = 0; i < mSize / 16; i += 1) {
        vmovups(ptr[regTmpYptr], Xbyak::Zmm(accumBaseIdx + i));
        lea(regTmpYptr, ptr[regTmpYptr + 16 * sizeof(float)]);
    }
    if (mLeft) {
        if (!maskType) {
            kmovw(k3,
                  ptr[stackPtr + offsetof(dlp::kernels::gemvParams, mr_mask)]);
        } else {
            kmovw(k3,
                  ptr[stackPtr + offsetof(dlp::kernels::gemvParams, m_mask)]);
        }
        vmovups(ptr[regTmpYptr] | k3, Xbyak::Zmm(accumBaseIdx + (mSize / 16)));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::storeYValuesRowStored<float>(int mSize)
{
    // Process each ZMM register (which contains 16 elements)
    for (int i = 0; i < (mSize + 15) / 16; i++) {
        int elements_in_zmm = (i < mSize / 16) ? 16 : (mSize % 16);
        if (elements_in_zmm == 0)
            break;

        // Extract 4 chunks of 128-bits (4 floats each) from the ZMM
        for (int j = 0; j < elements_in_zmm; j += 4) {
            vextractf32x4(Xbyak::Xmm(tmpBaseIdx + j / 4),
                          Xbyak::Zmm(accumBaseIdx + i), j / 4);
        }

        // Now store each extracted value to its proper row-strided location
        for (int j = 0; j < elements_in_zmm; j++) {
            int tmp_reg    = j / 4; // Which temp register has our value
            int pos_in_reg = j % 4; // Position within that temp register

            if (pos_in_reg == 0) {
                // First element in XMM can be stored directly
                vmovss(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg));
            } else {
                // Extract the 32-bit float to memory directly
                vpextrd(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg),
                        pos_in_reg);
            }

            // Move to next row
            add(regTmpYptr, regRsC);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::storeYValues<float>(int mSize, bool maskType)
{
    // Store values from Y
    mov(regTmpYptr, regYptr);
    if (cMatFormat == dlp::kernel_frame::storageFormat::colMajor) {
        RETURN_IF_ERROR((storeYValuesColStored<float>(mSize, maskType)));
    } else {
        RETURN_IF_ERROR((storeYValuesRowStored<float>(mSize)));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::process8RowBlock<float, float, float>(int mSize, bool isFringe)
{
    // Calculate number of full 8-row blocks
    int full_blocks = mSize / 8;
    int remaining   = mSize % 8;

    // Handle full blocks without conditionals
    for (int block = 0; block < full_blocks; block++) {
        // Process all 8 rows in the block
        for (int i = 0; i < 8; i++) {
            RETURN_IF_ERROR(
                (computeLoadFMA<float, float, float>(block * 8 + i, isFringe)));
        }

        // Update base pointer for next block
        lea(regTmpAptr, ptr[regTmpAptr + regRsA * 8]);
    }

    // Handle remaining rows
    if (remaining > 0) {
        for (int i = 0; i < remaining; i++) {
            RETURN_IF_ERROR((computeLoadFMA<float, float, float>(
                full_blocks * 8 + i, isFringe)));
        }
        if ((remaining == 1) || (remaining == 2) || (remaining == 4)) {
            lea(regTmpAptr, ptr[regTmpAptr + regRsA * remaining]);
        } else if ((remaining == 3)) {
            lea(regTmpAptr, ptr[regTmpAptr + regTmp1]);
        } else if ((remaining == 5)) {
            lea(regTmpAptr, ptr[regTmpAptr + regTmp2]);
        } else if ((remaining == 6)) {
            lea(regTmpAptr, ptr[regTmpAptr + regTmp1 * 2]);
        } else if ((remaining == 7)) {
            lea(regTmpAptr, ptr[regTmpAptr + regTmp3]);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::generateKernel<
    dlp::kernel_frame::kernelDatatype::f32f32f32of32>(
    const utils::gemvGeneratorParams& params)
{
    using aType     = float;
    using xType     = float;
    using yType     = float;
    using accumType = float;

    Xbyak::util::StackFrame frame(this, 1, 13, 0);
    initializeStackFrame(frame);

    initializeParameters<aType, xType, yType>(params);
    RETURN_IF_ERROR((allocateRegisters<accumType>()));

    // Acquire the addresses of A and Y
    mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, a)]);
    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, y)]);

    inLocalLabel();

    // Set the for-loop sequence for m-dimension
    if (params.mloop) {
        mov(regMIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvParams, m_iter)]);
        test(regMIter, regMIter);
        jz(label_m_loop_end, T_NEAR);
        L(label_m_loop_start);
        // }

        // Zero out accumulator registers for this m iteration
        regInitZmm(accumBaseIdx, MR);

        // // Y prefetch, before the k-loop
        // if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
        //     prefetcht0(ptr[regYptr]);
        // }

        // K-loop is not needed if alpha is zero
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Pre-calculate useful multiples of rsA
            lea(regTmp1,
                ptr[regRsA + regRsA * 2]); // regTmp1 = rsA + 2*rsA = 3*rsA
            lea(regTmp2,
                ptr[regRsA + regRsA * 4]); // regTmp2 = rsA + 4*rsA = 5*rsA
            lea(regTmp3,
                ptr[regTmp2 + regRsA * 2]); // regTmp3 = 5*rsA + 2*rsA = 7*rsA

            // Acquire the pointers for A
            // One is used in the m-loop, while other in the k-loop
            mov(regTmpAptr, regAptr);
            mov(regTmpYptr, regAptr);

            // Acquire the address of X
            mov(regXptr, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, x)]);

            // Set the for-loop sequence for k-dimension
            if (params.kloop) {
                mov(regKIter,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvParams, k_iter)]);
                test(regKIter, regKIter);
                jz(label_m_loop_k_loop_end, T_NEAR);
                L(label_m_loop_k_loop_start);

                // Load the X vector
                RETURN_IF_ERROR((loadXValues<xType>()));

                // Process all rows including fringe
                RETURN_IF_ERROR(
                    (process8RowBlock<aType, xType, accumType>(MR)));

                // Save current A pointer and update pointers for next k
                // iteration
                lea(regTmpYptr, ptr[regTmpYptr + regCsA * 8]);
                lea(regTmpYptr, ptr[regTmpYptr + regCsA * 8]);
                mov(regTmpAptr, regTmpYptr);
                add(regXptr, RegBytes); // Since B will be unit-strided

                dec(regKIter);
                jnz(label_m_loop_k_loop_start, T_NEAR);
            }
            L(label_m_loop_k_loop_end);

            if (params.kfringe) {
                mov(regKIter,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvParams, k_left)]);
                test(regKIter, regKIter);
                jz(label_m_loop_k_fringe_end, T_NEAR);
                L(label_m_loop_k_fringe_start);

                kmovw(
                    k3,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvParams, k_mask)]);
                RETURN_IF_ERROR((loadXValues<xType>(params.kfringe)));
                RETURN_IF_ERROR(
                    (process8RowBlock<aType, xType, accumType>(MR, true)));
            }
            L(label_m_loop_k_fringe_end);

            // Reduce the accumulation registers to XMMs, and put it in
            // ZMMs
            reduceAccumulation<accumType>(MR);

            // Alpha scaling
            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::one) {
                scaleAccumulationWithAlpha<accumType>(MR);
            }
        }

        // Working good for element-wise loads/stores for C.
        scaleYWithBeta<float>(MR, false);
        storeYValues<float>(MR, false);

        // if (params.mloop) {
        // Update pointers for next m iteration(for A and y)
        mov(regTmp2, MR);
        imul(regTmp2, regRsA);
        add(regAptr, regTmp2);
        mov(regTmp1, MR);
        imul(regTmp1, regRsC);
        add(regYptr, regTmp1);

        dec(regMIter);
        jnz(label_m_loop_start, T_NEAR);
    }

    L(label_m_loop_end);
    if (params.mfringe) {
        mov(regMIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvParams, m_left)]);
        test(regMIter, regMIter);
        jz(label_m_fringe_end, T_NEAR);
        L(label_m_fringe_start);

        regInitZmm(accumBaseIdx, M_LEFT);

        // K-loop is not needed if alpha is zero
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Pre-calculate useful multiples of rsA
            lea(regTmp1,
                ptr[regRsA + regRsA * 2]); // regTmp1 = rsA + 2*rsA = 3*rsA
            lea(regTmp2,
                ptr[regRsA + regRsA * 4]); // regTmp2 = rsA + 4*rsA = 5*rsA
            lea(regTmp3,
                ptr[regTmp2 + regRsA * 2]); // regTmp3 = 5*rsA + 2*rsA = 7*rsA

            // Acquire the pointers for A
            mov(regTmpAptr, regAptr);
            mov(regTmpYptr, regAptr);

            // Acquire the address of X
            mov(regXptr, ptr[stackPtr + offsetof(dlp::kernels::gemvParams, x)]);

            // Set the for-loop sequence for k-dimension
            if (params.kloop) {
                mov(regKIter,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvParams, k_iter)]);
                test(regKIter, regKIter);
                jz(label_m_fringe_k_loop_end, T_NEAR);
                L(label_m_fringe_k_loop_start);

                // Load the X vector
                RETURN_IF_ERROR((loadXValues<xType>()));

                // Process all rows including fringe
                RETURN_IF_ERROR(
                    (process8RowBlock<aType, xType, accumType>(M_LEFT)));

                // Update pointers for next k iteration
                lea(regTmpYptr, ptr[regTmpYptr + regCsA * 8]);
                lea(regTmpYptr, ptr[regTmpYptr + regCsA * 8]);
                mov(regTmpAptr, regTmpYptr);
                add(regXptr, RegBytes); // Since B will be unit-strided

                dec(regKIter);
                jnz(label_m_fringe_k_loop_start, T_NEAR);
            }
            L(label_m_fringe_k_loop_end);
            if (params.kfringe) {
                mov(regKIter,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvParams, k_left)]);
                test(regKIter, regKIter);
                jz(label_m_fringe_k_fringe_end, T_NEAR);
                L(label_m_fringe_k_fringe_start);

                kmovw(
                    k3,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvParams, k_mask)]);
                RETURN_IF_ERROR((loadXValues<xType>(true)));

                RETURN_IF_ERROR(
                    (process8RowBlock<aType, xType, accumType>(M_LEFT, true)));
            }
            L(label_m_fringe_k_fringe_end);

            // Reduce the accumulation registers to XMMs, and put it in
            // ZMMs
            reduceAccumulation<accumType>(M_LEFT);
            // Alpha scaling
            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::one) {
                scaleAccumulationWithAlpha<accumType>(M_LEFT);
            }
        }

        scaleYWithBeta<float>(M_LEFT, true);
        storeYValues<float>(M_LEFT, true);
    }
    // Might need more labels for the fringe cases.
    L(label_m_fringe_end);
    outLocalLabel();

    // Hack : Use the JIT generator(jitAVX512FP32) to check for NR = 1,
    //        and create a GEMV backend instance.
    //        Set utils::gemvGeneratorParams for main kernel(MR=16), then
    //        generate and dump the code. Generate the kerel through
    //        framework(GEMM/GEMV) to get the final dump.

    return dlp::jit::jitGeneratorError::success;
}
} // namespace amdzen::avx512gen
