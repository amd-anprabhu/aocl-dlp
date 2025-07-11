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

#include "kernel_register/kernel_register.hh"
#include "cpu_utils/cpu_features.hh"

namespace dlp::kernel_frame {

std::mutex         replacedKernelSink::mtx;
std::vector<void*> replacedKernelSink::valueSink;
std::set<void*>    replacedKernelSink::valueSet;

template<typename KERN_DISPATCH_TABLE>
kernelFrameError
kernelRegister<KERN_DISPATCH_TABLE>::registerGemmKernel(
    std::unique_ptr<kernels::kernelBase> _kB, std::string&& kernelFamily)
{
    if (!_kB) {
        return kernelFrameError::failure;
    }

    // Check if the kernel is compatible with the CPU features.
    std::vector<cpu_utils::isaFeature>& reqFeatures =
        _kB->getIsaFeaturesForKernel();
    auto hasFeatures =
        cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    if (!hasFeatures) {
        return kernelFrameError::failure;
    }

    kernels::kernelBase*         kB      = _kB.release();
    kernelInfo*                  kI      = kB->getKernelInfo();
    std::vector<kernelDatatype>& kDTypes = kB->getKernelDatatypes();

    auto routineIdx = utils::getUnderlyingValueOfEnum(kernelRoutineType::gemm);
    for (auto ele : kDTypes) {
        auto idx = utils::getUnderlyingValueOfEnum(ele);
        static_cast<void>(
            vecKDTs[routineIdx][idx]
                .template insert<gemmHashKeyGetter, gemmKeyComparator,
                                 kernelInfo, replacedKernelSink>(kI, kB));
    }

    return kernelFrameError::success;
}

template<typename KERN_DISPATCH_TABLE>
kernelBaseRef
kernelRegister<KERN_DISPATCH_TABLE>::registerAndGetGemmKernel(
    std::unique_ptr<kernels::kernelBase> _kB,
    std::string&&                        kernelFamily,
    kernelDatatype                       kDtype)
{
    if (!_kB) {
        return kernelBaseRef(nullptr);
    }

    // Check if the kernel is compatible with the CPU features.
    std::vector<cpu_utils::isaFeature>& reqFeatures =
        _kB->getIsaFeaturesForKernel();
    auto hasFeatures =
        cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    if (!hasFeatures) {
        return kernelBaseRef(nullptr);
    }

    kernels::kernelBase* kB = _kB.release();

    kernelInfo* kI = kB->getKernelInfo();

    auto routineIdx = utils::getUnderlyingValueOfEnum(kernelRoutineType::gemm);
    auto idx        = utils::getUnderlyingValueOfEnum(kDtype);

    // Safe to cast the voidFunctorPtr to kernelBase* because it is guaranteed
    // that kernelBase* was typecasted to voidFunctorPtr in insert operation.
    auto kernPtr = reinterpret_cast<kernels::kernelBase*>(
        vecKDTs[routineIdx][idx]
            .template insert<gemmHashKeyGetter, gemmKeyComparator, kernelInfo,
                             replacedKernelSink>(kI, kB));

    if (!kernPtr) {
        // TODO: Add logging here.
        return kernelBaseRef(nullptr);
    }

    return kernelBaseRef(kernPtr);
}

template<typename KERN_DISPATCH_TABLE>
kernelBaseRef
kernelRegister<KERN_DISPATCH_TABLE>::getGemmKernel(kernelInfo*    kI,
                                                   kernelDatatype kDtype)
{
    auto routineIdx = utils::getUnderlyingValueOfEnum(kernelRoutineType::gemm);
    auto dtypeIdx   = utils::getUnderlyingValueOfEnum(kDtype);

    // Safe to cast the voidFunctorPtr to kernelBase* because it is guaranteed
    // that kernelBase* was typecasted to voidFunctorPtr in insert operation.
    auto kB = reinterpret_cast<kernels::kernelBase*>(
        vecKDTs[routineIdx][dtypeIdx]
            .template query<gemmHashKeyGetter, gemmKeyComparator>(kI));

    // Cannot throw an exception here because this function is called by the
    // user and its not necessary to have the kernel inserted by the time query
    // is called.
    if (!kB) {
        // TODO: Add logging here.
        return kernelBaseRef(nullptr);
    }

    return kernelBaseRef(kB);
}

template class kernelRegister<
    ThreadSafeChainedDispatchTable<dlpKernelRegisterTableSize,
                                   dlpKernelRegisterTableNumBuckets>>;

} // namespace dlp::kernel_frame
