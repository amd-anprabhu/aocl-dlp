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

#include <optional>
#include <set>
#include <vector>

#include "de_backend.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "utils/type_utils.hh"

namespace dlp::de {

class decisionEngine
{
    void registerDecisionEngine()
    {
        // Only support F32 decision engine for now.
        auto kTypeIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelRoutineType::gemm);
        auto dtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::f32f32f32of32);
        backends[kTypeIdx][dtIdx] = new gemmF32DEBackend;
    }

    decisionEngine()
    {
        backends.reserve(
            static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                dlp::kernel_frame::kernelRoutineType::max_kernel_routines)));
        backends.resize(
            static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                dlp::kernel_frame::kernelRoutineType::max_kernel_routines)));

        for (auto& ele : backends) {
            ele.reserve(
                static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                    dlp::kernel_frame::kernelDatatype::max_kernel_datatypes)));
            ele.resize(
                static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                    dlp::kernel_frame::kernelDatatype::max_kernel_datatypes)),
                nullptr);
        }

        registerDecisionEngine();
    }

    ~decisionEngine()
    {
        // Using a set to avoid deleting the same value multiple times.
        // This is required since the same VALUE_TYPE could be inserted
        // multiple times depending on its usage by the composing class.
        // For example, a kernel could be registered multiple time with
        // different kernelDatatypes.
        std::set<iDEBackend*> valueSet;
        for (auto& ele : backends) {
            for (auto& ele2 : ele) {
                if (valueSet.count(ele2) == 0) {
                    valueSet.insert(ele2);
                    delete ele2;
                }
            }
        }
    }

    decisionEngine(const decisionEngine&)            = delete;
    decisionEngine& operator=(const decisionEngine&) = delete;
    decisionEngine(decisionEngine&&)                 = delete;
    decisionEngine& operator=(decisionEngine&&)      = delete;

    // 1-D array of dispatch tables: vecKDTs[routine_type][datatype]
    std::vector<std::vector<iDEBackend*>> backends;

  public:
    static decisionEngine& instance()
    {
        static decisionEngine de;
        return de;
    }

    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput*                            in,
        dlp::kernel_frame::kernelRoutineType kType,
        dlp::kernel_frame::kernelDatatype    dt)
    {
        auto kTypeIdx = utils::getUnderlyingValueOfEnum(kType);
        auto dtIdx    = utils::getUnderlyingValueOfEnum(dt);
        if (backends[kTypeIdx][dtIdx] != nullptr) {
            return backends[kTypeIdx][dtIdx]->getKernelInfoForInput(in);
        }
        return std::nullopt;
    }
};

inline decisionEngine&
decisionEngineInstance()
{
    return decisionEngine::instance();
}

} // namespace dlp::de
