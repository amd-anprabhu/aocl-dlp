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

#include <stack>

#if DLP_OS_WINDOWS
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "jit/jit_generator_base.hh"
#include "jit/xbyak/xbyak.h"

// Error handling macro to reduce repetitive code
#define RETURN_IF_ERROR(expr)                                                  \
    do {                                                                       \
        auto err = (expr);                                                     \
        if (err != dlp::jit::jitGeneratorError::success) {                     \
            return err;                                                        \
        }                                                                      \
    } while (0);

namespace amdzen::utils {

class jitHelperUtils
{
  public:
    // Function to dump JIT code to a file for debugging purposes. This
    // function will create a file with the name <code_name>_<m>x<n>.bin".
    // The code will be dumped in binary format.
    static void dump_jit_code(
        const void* code, int code_size, const char* code_name, int m, int n)
    {
        if (code) {
            static int counter = 0;
#define MAX_FNAME_LEN 256
            char fname[MAX_FNAME_LEN + 1];
            // TODO (Roma): support prefix for code / linux perf dumps
            snprintf(fname, MAX_FNAME_LEN, "%s_%dx%d.bin", code_name, m, n);
            counter++;
            FILE* fp = fopen(fname, "wb+");
            // Failure to dump code is not fatal
            if (fp) {
                int unused = fwrite(code, code_size, 1, fp);
                fclose(fp);
            }
        }
#undef MAX_FNAME_LEN
    }

    static void* allocateJitMemory(std::size_t jitKernSize)
    {
        void* codeBuffer = nullptr;
#if DLP_OS_WINDOWS
        codeBuffer =
            VirtualAlloc(nullptr, jitKernSize, MEM_COMMIT | MEM_RESERVE,
                         PAGE_EXECUTE_READWRITE);
#else
        codeBuffer =
            mmap(nullptr, jitKernSize, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (codeBuffer == MAP_FAILED) {
            codeBuffer = nullptr;
        }
#endif
        return codeBuffer;
    }

    static void deallocateJitMemory(void* codeBuffer, std::size_t jitKernSize)
    {
        if (codeBuffer != nullptr) {
#if DLP_OS_WINDOWS
            VirtualFree(codeBuffer, 0, MEM_RELEASE);
#else
            munmap(codeBuffer, jitKernSize);
#endif
        }
    }
};

typedef void (*jit_kernel)(dlp::kernels::gemmParams*);

constexpr uint64_t JIT_KERNEL_SIZE = 8 * 4096;

enum class kernelInstrType : uint16_t
{
    none = 0,
    avx2_xmm_16_reg,
    avx2_ymm_16_reg,
    avx512_xmm_32_reg,
    avx512_ymm_32_reg,
    avx512_zmm_32_reg
};

struct generatorParams
{
    int MR; // This MR can be of either main kernel or fringe kernel
    int NR; // This NR can be of either main kernel or fringe kernel
    int K_UNROLL;
    // This will be used to generate NR + " < nElemsPerReg" kernels,
    // where NR is a multiple of nElemsPerReg including "0".
    bool            useMask;
    bool            mLoop; // This will be set to true only for the main kernel
    bool            is_beta_zero; // skip beta scaling if beta is 0
    bool            is_alpha_one; // skip alpha scaling if alpha is 1
    kernelInstrType kType;
    std::vector<dlp::kernel_frame::kernelOpsMetaData> kernelOps;

    generatorParams(md_t            _MR,
                    md_t            _NR,
                    int             _K_UNROLL,
                    bool            _useMask,
                    bool            _mLoop,
                    bool            _is_beta_zero = false,
                    bool            _is_alpha_one = false,
                    kernelInstrType _kType        = kernelInstrType::none)
        : MR(_MR)
        , NR(_NR)
        , K_UNROLL(_K_UNROLL)
        , useMask(_useMask)
        , mLoop(_mLoop)
        , is_beta_zero(_is_beta_zero)
        , is_alpha_one(_is_alpha_one)
        , kType(_kType)
    {
    }

    generatorParams(const generatorParams& other)
        : MR(other.MR)
        , NR(other.NR)
        , K_UNROLL(other.K_UNROLL)
        , useMask(other.useMask)
        , mLoop(other.mLoop)
        , is_beta_zero(other.is_beta_zero)
        , is_alpha_one(other.is_alpha_one)
        , kType(other.kType)
        , kernelOps(other.kernelOps)
    {
    }

    generatorParams& operator=(const generatorParams& other)
    {
        if (this != std::addressof(other)) {
            MR           = other.MR;
            NR           = other.NR;
            K_UNROLL     = other.K_UNROLL;
            useMask      = other.useMask;
            mLoop        = other.mLoop;
            is_beta_zero = other.is_beta_zero;
            is_alpha_one = other.is_alpha_one;
            kType        = other.kType;
            kernelOps    = other.kernelOps;
        }
        return *this;
    }

    generatorParams(generatorParams&& other)
        : MR(other.MR)
        , NR(other.NR)
        , K_UNROLL(other.K_UNROLL)
        , useMask(other.useMask)
        , mLoop(other.mLoop)
        , is_beta_zero(other.is_beta_zero)
        , is_alpha_one(other.is_alpha_one)
        , kType(other.kType)
        , kernelOps(std::move(other.kernelOps))
    {
    }

    generatorParams& operator=(generatorParams&& other)
    {
        if (this != std::addressof(other)) {
            MR           = other.MR;
            NR           = other.NR;
            K_UNROLL     = other.K_UNROLL;
            useMask      = other.useMask;
            mLoop        = other.mLoop;
            is_beta_zero = other.is_beta_zero;
            is_alpha_one = other.is_alpha_one;
            kType        = other.kType;
            kernelOps    = std::move(other.kernelOps);
        }
        return *this;
    }

    ~generatorParams() = default;
};

template<typename REG_TYPE>
class registerGuard
{
    Xbyak::CodeGenerator* jit;
    std::stack<REG_TYPE>  regStack;

  public:
    registerGuard(Xbyak::CodeGenerator* _jit)
        : jit{ _jit }
    {
    }

    void saveRegister(REG_TYPE reg)
    {
        regStack.push(reg);
        jit->push(reg);
    }

    ~registerGuard()
    {
        while (!regStack.empty()) {
            auto tReg = regStack.top();
            jit->pop(tReg);
            regStack.pop();
        }
    }

    registerGuard(registerGuard&)             = delete;
    registerGuard(registerGuard&&)            = delete;
    registerGuard& operator=(registerGuard&)  = delete;
    registerGuard& operator=(registerGuard&&) = delete;
};

} // namespace amdzen::utils
