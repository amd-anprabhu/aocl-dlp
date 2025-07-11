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

namespace dlp::utils {

template<typename T, typename PARAM_TYPE, typename ERROR_TYPE>
class ptrWrapper
{
    T* kB;

  public:
    ptrWrapper(T* kB)
        : kB(kB)
    {
    }
    ~ptrWrapper() { kB = nullptr; }

    ptrWrapper(const ptrWrapper& other)            = delete;
    ptrWrapper& operator=(const ptrWrapper& other) = delete;

    ptrWrapper(ptrWrapper&& other)
        : kB(other.kB)
    {
        other.kB = nullptr;
    }

    ptrWrapper& operator=(ptrWrapper&& other)
    {
        auto temp = kB;
        kB        = other.kB;
        other.kB  = temp;
        return *this;
    }

    void reset()
    {
        if (kB) {
            kB = nullptr;
        }
    }

    ERROR_TYPE operator()(PARAM_TYPE kP) { return kB->operator()(kP); }

    [[nodiscard]] bool isValid() const { return kB != nullptr; }

    [[nodiscard]] explicit operator bool() const { return isValid(); }

    [[nodiscard]] T* operator->() const { return kB; }

    T* getPtr() const { return kB; }
};

} // namespace dlp::utils
