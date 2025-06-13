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

#include <cstdint>
#include <ostream>

/**
 * @namespace dlp
 * @brief Deep Learning Primitives namespace
 */
namespace dlp {

/**
 * @namespace dlp::testing
 * @brief Testing framework for Deep Learning Primitives
 */
namespace testing {

    /**
     * @enum MatrixType
     * @brief Enumeration of supported matrix data types
     */
    enum class MatrixType : uint16_t
    {
        u4 = 0, ///< Unsigned 4-bit integer
        u8,     ///< Unsigned 8-bit integer
        u16,    ///< Unsigned 16-bit integer
        u32,    ///< Unsigned 32-bit integer
        s4,     ///< Signed 4-bit integer
        s8,     ///< Signed 8-bit integer
        s16,    ///< Signed 16-bit integer
        s32,    ///< Signed 32-bit integer
        f32,    ///< 32-bit floating point
        bf16,   ///< Brain floating point 16-bit
    };

    /**
     * @brief Stream output operator for MatrixType
     * @param os Output stream
     * @param type MatrixType to output
     * @return Reference to the output stream
     */
    inline std::ostream& operator<<(std::ostream& os, MatrixType type)
    {
        switch (type) {
            case MatrixType::u4:
                return os << "u4";
            case MatrixType::u8:
                return os << "u8";
            case MatrixType::u16:
                return os << "u16";
            case MatrixType::u32:
                return os << "u32";
            case MatrixType::s4:
                return os << "s4";
            case MatrixType::s8:
                return os << "s8";
            case MatrixType::s16:
                return os << "s16";
            case MatrixType::s32:
                return os << "s32";
            case MatrixType::f32:
                return os << "f32";
            case MatrixType::bf16:
                return os << "bf16";
            default:
                return os << "unknown";
        }
    }

    /**
     * @enum MatrixLayout
     * @brief Enumeration of supported matrix memory layouts
     */
    enum class MatrixLayout
    {
        ROW_MAJOR,   ///< Row-major layout (C/C++ style)
        COLUMN_MAJOR ///< Column-major layout (Fortran style)
    };

    /**
     * @brief Stream output operator for MatrixLayout
     * @param os Output stream
     * @param layout MatrixLayout to output
     * @return Reference to the output stream
     */
    inline std::ostream& operator<<(std::ostream& os, MatrixLayout layout)
    {
        switch (layout) {
            case MatrixLayout::ROW_MAJOR:
                return os << "row-major";
            case MatrixLayout::COLUMN_MAJOR:
                return os << "column-major";
            default:
                return os << "unknown";
        }
    }

    /**
     * @struct data_type
     * @brief Type container for matrix data with pointer management
     * @tparam T Data type of the elements
     */
    template<typename T>
    struct data_type
    {
        T* data; ///< Pointer to the data array
    };

    // Type aliases for various data types
    using u8  = data_type<uint8_t>;  ///< Alias for unsigned 8-bit data
    using u16 = data_type<uint16_t>; ///< Alias for unsigned 16-bit data
    using u32 = data_type<uint32_t>; ///< Alias for unsigned 32-bit data
    using s8  = data_type<int8_t>;   ///< Alias for signed 8-bit data
    using s16 = data_type<int16_t>;  ///< Alias for signed 16-bit data
    using s32 = data_type<int32_t>;  ///< Alias for signed 32-bit data
    using f32 = data_type<float>;    ///< Alias for 32-bit floating point data
    using bf16 =
        data_type<uint16_t>; ///< Alias for BF16 data (stored as uint16_t)

    /**
     * @struct s4
     * @brief Special handling for signed 4-bit data (packed into int8_t)
     */
    struct s4 : data_type<int8_t>
    {};

    /**
     * @struct u4
     * @brief Special handling for unsigned 4-bit data (packed into uint8_t)
     */
    struct u4 : data_type<uint8_t>
    {};

    /**
     * @struct MatrixData
     * @brief Container for matrix data with type-based storage
     *
     * This structure provides a union-based approach to storing different data
     * types while maintaining type information.
     */
    struct MatrixData
    {
        MatrixType type; ///< The type of data stored in the matrix

        /**
         * @brief Union for storing different data types
         */
        union
        {
            u4   u4_data;   ///< Storage for unsigned 4-bit data
            u8   u8_data;   ///< Storage for unsigned 8-bit data
            u16  u16_data;  ///< Storage for unsigned 16-bit data
            u32  u32_data;  ///< Storage for unsigned 32-bit data
            s4   s4_data;   ///< Storage for signed 4-bit data
            s8   s8_data;   ///< Storage for signed 8-bit data
            s16  s16_data;  ///< Storage for signed 16-bit data
            s32  s32_data;  ///< Storage for signed 32-bit data
            f32  f32_data;  ///< Storage for 32-bit floating point data
            bf16 bf16_data; ///< Storage for BF16 data
        };

        /**
         * @brief Get raw pointer to the matrix data
         *
         * @return void* Pointer to the matrix data (type-erased)
         */
        void* getMatrixPtr() const
        {
            switch (type) {
                case MatrixType::u4:
                    return u4_data.data;
                case MatrixType::u8:
                    return u8_data.data;
                case MatrixType::u16:
                    return u16_data.data;
                case MatrixType::u32:
                    return u32_data.data;
                case MatrixType::s4:
                    return s4_data.data;
                case MatrixType::s8:
                    return s8_data.data;
                case MatrixType::s16:
                    return s16_data.data;
                case MatrixType::s32:
                    return s32_data.data;
                case MatrixType::f32:
                    return f32_data.data;
                case MatrixType::bf16:
                    return bf16_data.data;
                default:
                    return nullptr;
            }
        }
    };

} // namespace testing
} // namespace dlp
