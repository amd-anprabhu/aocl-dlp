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

#ifndef AVX512_GEMV_HH
#define AVX512_GEMV_HH

#include "jit/jit_generator_base.hh"
#include "jit/xbyak/xbyak.h"
#include "jit/xbyak/xbyak_util.h"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_handler.hh"
#include "traits.hh"

namespace amdzen::avx512gen {

class jitAVX512GEMVN1 : public Xbyak::CodeGenerator
{
  private:
    // Constants(Future scope is to see if these can be abstracted to a parent
    // class to set AVX512 ISA features)
    static constexpr int RegBytes = 64; // Size of ZMM register in bytes
    static constexpr int numRegs  = 32; // Number of ZMM registers

    // Dimensions
    int                              MR; // Number of rows to process at once
    int                              M_LEFT;  // M-dimension left over elements
    dlp::kernel_frame::storageFormat yFormat; // Storage format of C matrix
    dlp::kernel_frame::scalingType alphaScalingType; // Type of kernel operation
    dlp::kernel_frame::scalingType betaScalingType;  // Type of beta scaling

    // Register counts and indices
    int yReg;     // Number of registers for loading/storing from Y
    int aReg;     // Number of registers for matrix A
    int xReg;     // Number of registers for vector x
    int accumReg; // Number of registers for accumulation (partial dot products)
    int tmpReg;   // Number of registers for temporary use
    int yBaseIdx; // Starting index for accumulation registers (from end)
    int aBaseIdx; // Starting index for A registers (from beginning)
    int xBaseIdx; // Starting index for x registers (after A registers)
    int accumBaseIdx; // Starting index for accumulation registers (after A and
                      // x
    int tmpBaseIdx;   // Starting index for temporary registers (after A and x
    // Matrix/Vector pointers and strides
    Xbyak::Reg64 stackPtr;            // Stack frame pointer
    Xbyak::Reg64 regAptr, regTmpAptr; // Pointer to matrix A and its temp
    Xbyak::Reg64 regXptr;             // Pointer to vector x
    Xbyak::Reg64 regYptr, regTmpYptr; // Pointer to vector y and its temp
    Xbyak::Reg64 regRsA;              // Row stride for A
    Xbyak::Reg64 regCsA;              // Column stride for A
    Xbyak::Reg64 regRsC;              // Row stride for C
    Xbyak::Reg64 regMIter;            // M-loop iterator
    Xbyak::Reg64 regKIter;            // K-loop iterator
    Xbyak::Reg64 regTmp1;             // General purpose temporary register 1
    Xbyak::Reg64 regTmp2;             // General purpose temporary register 2
    Xbyak::Reg64 regTmp3;             // General purpose temporary register 3

    // Labels for code sections
    Xbyak::Label label_m_loop_start;            // Main m-dimension loop
    Xbyak::Label label_m_loop_end;              // End of m-dimension loop
    Xbyak::Label label_m_loop_k_loop_start;     // Main k-dimension loop
    Xbyak::Label label_m_loop_k_loop_end;       // End of k-dimension loop
    Xbyak::Label label_m_fringe_k_loop_start;   // Main k-dimension loop
    Xbyak::Label label_m_fringe_k_loop_end;     // End of k-dimension loop
    Xbyak::Label label_m_fringe_start;          // Handle m-dimension remainder
    Xbyak::Label label_m_fringe_end;            // End of m-dimension remainder
    Xbyak::Label label_m_loop_k_fringe_start;   // Handle k-dimension remainder
    Xbyak::Label label_m_loop_k_fringe_end;     // End of k-dimension remainder
    Xbyak::Label label_m_fringe_k_fringe_start; // Handle k-dimension remainder
    Xbyak::Label label_m_fringe_k_fringe_end;   // End of k-dimension remainder
    Xbyak::Label label_reduce_start;            // Reduction operations
    Xbyak::Label label_store_result;            // Store final results

    // Stack frame management
    void initializeStackFrame(Xbyak::util::StackFrame&);

    // Register initialization
    void regInitZmm(int, int);

    // Implementation utilities
    template<typename accumType>
    dlp::jit::jitGeneratorError allocateRegisters();

    template<typename aType, typename xType, typename yType>
    void initializeParameters(const utils::gemvN1GeneratorParams&);

    // Core computation functions
    template<typename aType>
    dlp::jit::jitGeneratorError loadAValues(int, int, bool = false);

    template<typename xType>
    dlp::jit::jitGeneratorError loadXValues(bool = false);

    template<typename yType>
    dlp::jit::jitGeneratorError loadYValues(int);

    template<typename aType, typename xType, typename accumType>
    dlp::jit::jitGeneratorError computeFMA(int, int);

    template<typename aType, typename xType, typename accumType>
    dlp::jit::jitGeneratorError computeLoadFMA(int, bool = false);

    template<typename accumType>
    dlp::jit::jitGeneratorError reduce4ZMMtoXMM(int, int, int);

    template<typename accumType>
    dlp::jit::jitGeneratorError reduceAccumulation(int);

    template<typename accumType>
    dlp::jit::jitGeneratorError scaleAccumulationWithAlpha(int);

    template<typename yType>
    dlp::jit::jitGeneratorError scaleYWithBetaColStored(int,
                                                        bool = false,
                                                        bool = false);
    template<typename yType>
    dlp::jit::jitGeneratorError scaleYWithBetaRowStored(int, bool = false);

    template<typename yType>
    dlp::jit::jitGeneratorError scaleYWithBeta(int, bool = false);

    template<typename yType>
    dlp::jit::jitGeneratorError storeYValuesColStored(int, bool = false);

    template<typename yType>
    dlp::jit::jitGeneratorError storeYValuesRowStored(int);

    template<typename yType>
    dlp::jit::jitGeneratorError storeYValues(int, bool = false);

    template<typename aType, typename xType, typename accumType>
    dlp::jit::jitGeneratorError process8RowBlock(int, bool = false);

  public:
    // Enforcing RAII, disallowing copy/move operations
    jitAVX512GEMVN1(void* buffer = nullptr, size_t size = 0);
    ~jitAVX512GEMVN1()                            = default;
    jitAVX512GEMVN1(jitAVX512GEMVN1&)             = delete;
    jitAVX512GEMVN1& operator=(jitAVX512GEMVN1&)  = delete;
    jitAVX512GEMVN1(jitAVX512GEMVN1&&)            = delete;
    jitAVX512GEMVN1& operator=(jitAVX512GEMVN1&&) = delete;

    // Main kernel generation interface
    template<dlp::kernel_frame::kernelDatatype KDT>
    dlp::jit::jitGeneratorError generateKernel(
        const utils::gemvN1GeneratorParams& params);

    // Get the generated kernel function pointer
    // This class will also contain the pointer type to the JIT kernel
    // That way, this typedef is available only when an instance of this class
    // is created.
    // using jit_kernel = void (*)(dlp::kernels::gemvN1Params*);
    // jit_kernel getKernel() { return getCode<jit_kernel>(); }
};

class jitAVX512GEMVM1 : public Xbyak::CodeGenerator
{
  private:
    // Constants for AVX512 architecture
    static constexpr int RegBytes = 64; // Size of ZMM register in bytes
    static constexpr int numRegs  = 32; // Number of ZMM registers

    // Dimensions and configuration
    int NR; // Number of elements to process per iteration (typically 64)
    int KC; // K-dimension blocksize
    AOCL_MEMORY_TAG                  mtag_b;  // Memory tag for B matrix
    dlp::kernel_frame::storageFormat yFormat; // Storage format of output
    dlp::kernel_frame::scalingType   alphaScalingType; // Type of alpha scaling
    dlp::kernel_frame::scalingType   betaScalingType;  // Type of beta scaling

    // Register counts and indices
    int yReg;         // Number of registers for loading/storing from Y
    int bReg;         // Number of registers for matrix B
    int xReg;         // Number of registers for vector x
    int accumReg;     // Number of registers for accumulation
    int yBaseIdx;     // Starting index for accumulation registers
    int bBaseIdx;     // Starting index for B registers
    int xBaseIdx;     // Starting index for x registers
    int accumBaseIdx; // Starting index for accumulation registers

    // General purpose registers
    Xbyak::Reg64 stackPtr;            // Stack frame pointer
    Xbyak::Reg64 regTmpBptr;          // Pointer to matrix B (single row)
    Xbyak::Reg64 regXptr;             // Pointer to vector x
    Xbyak::Reg64 regYptr, regTmpYptr; // Pointer to scalar output y and its temp
    Xbyak::Reg64 regNIter;            // N-loop iterator
    Xbyak::Reg64 regKIter;            // K-loop iterator (for vectorization)
    Xbyak::Reg64 regKSubIter;         // K-loop left over elements
    Xbyak::Reg64 regRsB;              // Column stride for B
    Xbyak::Reg64 regPsB;              // Packed stride for B
    Xbyak::Reg64 regTmp1;             // Temporary register 1
    Xbyak::Reg64 regTmp2;             // Temporary register 2
    // Xbyak::Reg64 regTmp3;             // Temporary register 3
    Xbyak::Reg64 regIncN; // Increment for N-loop
    Xbyak::Reg64 regIncK; // Increment for K-loop

    // Labels for code sections
    Xbyak::Label label_n_loop_start;            // Vectorization loop
    Xbyak::Label label_n_loop_end;              // End of vectorization loop
    Xbyak::Label label_n_loop_k_loop_start;     // Main n-dimension loop
    Xbyak::Label label_n_loop_k_loop_end;       // End of n-dimension loop
    Xbyak::Label label_n_loop_k_fringe_start;   // Main n-dimension loop
    Xbyak::Label label_n_loop_k_fringe_end;     // End of n-dimension loop
    Xbyak::Label label_n_fringe_start;          // Handle n-dimension remainder
    Xbyak::Label label_n_fringe_end;            // End of n-dimension remainder
    Xbyak::Label label_n_fringe_k_loop_start;   // Handle n-dimension remainder
    Xbyak::Label label_n_fringe_k_loop_end;     // End of n-dimension remainder
    Xbyak::Label label_n_fringe_k_fringe_start; // Handle n-dimension remainder
    Xbyak::Label label_n_fringe_k_fringe_end;   // End of n-dimension remainder
    Xbyak::Label label_accumulate_result;       // Accumulate final result
    Xbyak::Label label_store_result;            // Store final scalar result

    // Temporary labels
    Xbyak::Label temp_label_1;
    Xbyak::Label temp_label_2;

    // Stack frame management
    void initializeStackFrame(Xbyak::util::StackFrame&);

    // Register initialization
    void regInitZmm(int baseIdx, int numRegs);

    // Implementation utilities
    template<typename accumType>
    dlp::jit::jitGeneratorError allocateRegisters();

    template<typename bType, typename xType, typename yType>
    void initializeParameters(const utils::gemvM1GeneratorParams&);

    // Core computation functions
    template<typename bType, typename xType, typename accumType>
    dlp::jit::jitGeneratorError compute4x64(bool = false);

    template<typename bType, typename xType, typename accumType>
    dlp::jit::jitGeneratorError compute1x64(bool = false);

    template<typename bType, typename xType, typename accumType>
    dlp::jit::jitGeneratorError loopKSubIter(bool = false, bool = false);

    template<typename bType, typename xType, typename yType>
    dlp::jit::jitGeneratorError finalAccumulate();

    template<typename bType, typename xType, typename yType>
    dlp::jit::jitGeneratorError scaleWithAlpha();

    template<typename yType>
    dlp::jit::jitGeneratorError scaleYWithBeta(bool = false);

    template<typename yType>
    dlp::jit::jitGeneratorError storeYValues(bool = false);

  public:
    // Enforcing RAII, disallowing copy/move operations
    jitAVX512GEMVM1(void* buffer = nullptr, size_t size = 0);
    ~jitAVX512GEMVM1()                            = default;
    jitAVX512GEMVM1(jitAVX512GEMVM1&)             = delete;
    jitAVX512GEMVM1& operator=(jitAVX512GEMVM1&)  = delete;
    jitAVX512GEMVM1(jitAVX512GEMVM1&&)            = delete;
    jitAVX512GEMVM1& operator=(jitAVX512GEMVM1&&) = delete;

    // Main kernel generation interface
    template<dlp::kernel_frame::kernelDatatype KDT>
    dlp::jit::jitGeneratorError generateKernel(
        const utils::gemvM1GeneratorParams& params);

    // Get the generated kernel function pointer
    // utils::jit_gemv_m1_kernel getKernel() {
    //     return getCode<utils::jit_gemv_m1_kernel>();
    // }
};

} // namespace amdzen::avx512gen

#endif
