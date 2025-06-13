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

/**
 * @file matrix.cc
 * @brief Implementation of the Matrix class
 *
 * This file contains the implementation of the Matrix class that handles
 * various data types with support for different memory layouts and virtual
 * transposition.
 */

#include "framework/matrix.hh"
#include "classic/dlp_base_types.h"
#include <chrono>  // For time-based seeding
#include <cmath>   // For std::abs
#include <cstring> // For std::memcpy
#include <random>  // For random number generation

namespace dlp { namespace testing {

    /**
     * @brief Enhanced constructor implementation with full parameterization
     *
     * Allocates memory for the matrix based on its type, size, and layout.
     *
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     * @param type Data type of the matrix elements
     * @param layout Memory layout (ROW_MAJOR or COLUMN_MAJOR)
     * @param leadingDim Leading dimension (0 for automatic calculation)
     * @param transposed Whether the matrix is logically transposed without data
     * movement
     */
    Matrix::Matrix(md_t         rows,
                   md_t         cols,
                   MatrixType   type,
                   MatrixLayout layout,
                   md_t         leadingDim,
                   bool         is_trans,
                   bool         is_reordered)
        : m_rows(rows)
        , m_cols(cols)
        , m_layout(layout)
        , m_transposed(is_trans)
        , m_reordered(is_reordered)
    {
        // Calculate leading dimension if not specified
        if (leadingDim == 0) {
            m_leadingDim = (layout == MatrixLayout::ROW_MAJOR) ? cols : rows;
        } else {
            m_leadingDim = leadingDim;
        }

        m_data.type    = type;
        md_t allocSize = getAllocSize();
        switch (type) {
            case MatrixType::u4:
                // For 4-bit types, we need half the bytes (2 values per byte)
                m_data.u4_data.data = new uint8_t[(allocSize + 1) / 2];
                break;
            case MatrixType::u8:
                m_data.u8_data.data = new uint8_t[allocSize];
                break;
            case MatrixType::u16:
                m_data.u16_data.data = new uint16_t[allocSize];
                break;
            case MatrixType::u32:
                m_data.u32_data.data = new uint32_t[allocSize];
                break;
            case MatrixType::s4:
                // For 4-bit types, we need half the bytes (2 values per byte)
                m_data.s4_data.data = new int8_t[(allocSize + 1) / 2];
                break;
            case MatrixType::s8:
                m_data.s8_data.data = new int8_t[allocSize];
                break;
            case MatrixType::s16:
                m_data.s16_data.data = new int16_t[allocSize];
                break;
            case MatrixType::s32:
                m_data.s32_data.data = new int32_t[allocSize];
                break;
            case MatrixType::f32:
                m_data.f32_data.data = new float[allocSize];
                break;
            case MatrixType::bf16:
                m_data.bf16_data.data = new uint16_t[allocSize];
                break;
            default:
                throw std::runtime_error("Invalid matrix type");
        }
    }

    /**
     * @brief Backward compatibility constructor implementation
     *
     * Delegates to the enhanced constructor with default parameters.
     *
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     * @param type Data type of the matrix elements
     */
    Matrix::Matrix(md_t rows, md_t cols, MatrixType type)
        : Matrix(rows, cols, type, MatrixLayout::ROW_MAJOR, 0, false)
    {
        // Implementation is delegated to the enhanced constructor
    }

    /**
     * @brief Destructor implementation
     *
     * Frees allocated memory based on the matrix type.
     */
    Matrix::~Matrix()
    {
        switch (m_data.type) {
            case MatrixType::u4:
                delete[] m_data.u4_data.data;
                break;
            case MatrixType::u8:
                delete[] m_data.u8_data.data;
                break;
            case MatrixType::u16:
                delete[] m_data.u16_data.data;
                break;
            case MatrixType::u32:
                delete[] m_data.u32_data.data;
                break;
            case MatrixType::s4:
                delete[] m_data.s4_data.data;
                break;
            case MatrixType::s8:
                delete[] m_data.s8_data.data;
                break;
            case MatrixType::s16:
                delete[] m_data.s16_data.data;
                break;
            case MatrixType::s32:
                delete[] m_data.s32_data.data;
                break;
            case MatrixType::f32:
                delete[] m_data.f32_data.data;
                break;
            case MatrixType::bf16:
                delete[] m_data.bf16_data.data;
                break;
            default:
                break;
        }
    }

    /**
     * @brief Copy assignment operator implementation
     *
     * Creates a deep copy of the source matrix.
     *
     * @param other The matrix to copy from
     * @return Matrix& Reference to this matrix
     */
    Matrix& Matrix::operator=(const Matrix& other)
    {
        // Self-assignment check
        if (this == &other) {
            return *this;
        }

        // Free existing resources
        switch (m_data.type) {
            case MatrixType::u4:
                delete[] m_data.u4_data.data;
                break;
            case MatrixType::u8:
                delete[] m_data.u8_data.data;
                break;
            case MatrixType::u16:
                delete[] m_data.u16_data.data;
                break;
            case MatrixType::u32:
                delete[] m_data.u32_data.data;
                break;
            case MatrixType::s4:
                delete[] m_data.s4_data.data;
                break;
            case MatrixType::s8:
                delete[] m_data.s8_data.data;
                break;
            case MatrixType::s16:
                delete[] m_data.s16_data.data;
                break;
            case MatrixType::s32:
                delete[] m_data.s32_data.data;
                break;
            case MatrixType::f32:
                delete[] m_data.f32_data.data;
                break;
            case MatrixType::bf16:
                delete[] m_data.bf16_data.data;
                break;
            default:
                break;
        }

        // Copy metadata
        m_rows       = other.m_rows;
        m_cols       = other.m_cols;
        m_layout     = other.m_layout;
        m_leadingDim = other.m_leadingDim;
        m_transposed = other.m_transposed;
        m_data.type  = other.m_data.type;

        // Get allocation size using the member function
        md_t allocSize = getAllocSize();

        // Allocate and copy data based on type
        switch (m_data.type) {
            case MatrixType::u4: {
                m_data.u4_data.data = new uint8_t[(allocSize + 1) / 2];
                std::memcpy(m_data.u4_data.data, other.m_data.u4_data.data,
                            (allocSize + 1) / 2);
                break;
            }
            case MatrixType::u8: {
                m_data.u8_data.data = new uint8_t[allocSize];
                std::memcpy(m_data.u8_data.data, other.m_data.u8_data.data,
                            allocSize);
                break;
            }
            case MatrixType::u16: {
                m_data.u16_data.data = new uint16_t[allocSize];
                std::memcpy(m_data.u16_data.data, other.m_data.u16_data.data,
                            allocSize * sizeof(uint16_t));
                break;
            }
            case MatrixType::u32: {
                m_data.u32_data.data = new uint32_t[allocSize];
                std::memcpy(m_data.u32_data.data, other.m_data.u32_data.data,
                            allocSize * sizeof(uint32_t));
                break;
            }
            case MatrixType::s4: {
                m_data.s4_data.data = new int8_t[(allocSize + 1) / 2];
                std::memcpy(m_data.s4_data.data, other.m_data.s4_data.data,
                            (allocSize + 1) / 2);
                break;
            }
            case MatrixType::s8: {
                m_data.s8_data.data = new int8_t[allocSize];
                std::memcpy(m_data.s8_data.data, other.m_data.s8_data.data,
                            allocSize);
                break;
            }
            case MatrixType::s16: {
                m_data.s16_data.data = new int16_t[allocSize];
                std::memcpy(m_data.s16_data.data, other.m_data.s16_data.data,
                            allocSize * sizeof(int16_t));
                break;
            }
            case MatrixType::s32: {
                m_data.s32_data.data = new int32_t[allocSize];
                std::memcpy(m_data.s32_data.data, other.m_data.s32_data.data,
                            allocSize * sizeof(int32_t));
                break;
            }
            case MatrixType::f32: {
                m_data.f32_data.data = new float[allocSize];
                std::memcpy(m_data.f32_data.data, other.m_data.f32_data.data,
                            allocSize * sizeof(float));
                break;
            }
            case MatrixType::bf16: {
                m_data.bf16_data.data = new uint16_t[allocSize];
                std::memcpy(m_data.bf16_data.data, other.m_data.bf16_data.data,
                            allocSize * sizeof(uint16_t));
                break;
            }
            default:
                throw std::runtime_error(
                    "Invalid matrix type in copy assignment");
        }

        return *this;
    }

    /**
     * @brief Get the matrix data container
     *
     * @return MatrixData structure containing the matrix data and type
     */
    MatrixData Matrix::getMatrixData() const
    {
        return m_data;
    }

    /**
     * @brief Get the matrix data type
     *
     * @return MatrixType enum value indicating the data type
     */
    MatrixType Matrix::getMatrixType() const
    {
        return m_data.type;
    }

    /**
     * @brief Get the number of rows in the matrix
     *
     * @return md_t Number of rows
     */
    md_t Matrix::getRows() const
    {
        return m_rows;
    }

    /**
     * @brief Get the number of columns in the matrix
     *
     * @return md_t Number of columns
     */
    md_t Matrix::getCols() const
    {
        return m_cols;
    }

    /**
     * @brief Get the matrix memory layout
     *
     * @return MatrixLayout enum value (ROW_MAJOR or COLUMN_MAJOR)
     */
    MatrixLayout Matrix::getLayout() const
    {
        return m_layout;
    }

    /**
     * @brief Check if the matrix is logically transposed
     *
     * @return bool True if the matrix is transposed
     */
    bool Matrix::isTransposed() const
    {
        return m_transposed;
    }

    /**
     * @brief Check if the matrix is reordered
     *
     * @return bool True if the matrix is reordered
     */
    bool Matrix::isReordered() const
    {
        return m_reordered;
    }

    /**
     * @brief Get the leading dimension of the matrix
     *
     * @return md_t The leading dimension (stride between rows or columns)
     */
    md_t Matrix::getLeadingDimension() const
    {
        return m_leadingDim;
    }

    /**
     * @brief Get the effective number of rows after considering transposition
     *
     * @return md_t Effective number of rows (columns if transposed)
     */
    md_t Matrix::getEffectiveRows() const
    {
        return m_transposed ? m_cols : m_rows;
    }

    /**
     * @brief Get the effective number of columns after considering
     * transposition
     *
     * @return md_t Effective number of columns (rows if transposed)
     */
    md_t Matrix::getEffectiveCols() const
    {
        return m_transposed ? m_rows : m_cols;
    }

    /**
     * @brief Get the allocation size for the matrix data
     *
     * Calculates the total number of elements allocated for the matrix
     * based on the leading dimension and layout.
     *
     * @return md_t The allocation size in number of elements
     */
    md_t Matrix::getAllocSize() const
    {
        if (m_layout == MatrixLayout::ROW_MAJOR) {
            return m_rows * m_leadingDim;
        } else {
            return m_cols * m_leadingDim;
        }
    }

    /**
     * @brief Set the reordering flag
     *
     * @param reordered Whether the matrix is reordered
     */
    void Matrix::setReordered(bool reordered)
    {
        m_reordered = reordered;
    }

    /**
     * @brief Compare two matrices for equality
     *
     * Checks if dimensions, type, and content match.
     *
     * @param other The matrix to compare with
     * @return bool True if matrices are equal, false otherwise
     */
    bool Matrix::operator==(const Matrix& other) const
    {

        // Check if dimensions match
        if (getEffectiveRows() != other.getEffectiveRows()
            || getEffectiveCols() != other.getEffectiveCols()) {
            return false;
        }

        // Check if types match
        if (m_data.type != other.m_data.type) {
            return false;
        }

        // Compare data based on type
        md_t rows = getEffectiveRows();
        md_t cols = getEffectiveCols();

        // For simplicity, we compare element by element
        // In a real implementation, you might optimize this for performance
        for (md_t i = 0; i < rows; i++) {
            for (md_t j = 0; j < cols; j++) {
                // Get indices based on layout and transposition
                md_t this_idx, other_idx;

                // Handle different layouts and transposition for this matrix
                if (m_layout == MatrixLayout::ROW_MAJOR) {
                    this_idx = m_transposed ? j * m_leadingDim + i
                                            : i * m_leadingDim + j;
                } else {
                    this_idx = m_transposed ? i * m_leadingDim + j
                                            : j * m_leadingDim + i;
                }

                // Handle different layouts and transposition for other matrix
                if (other.m_layout == MatrixLayout::ROW_MAJOR) {
                    other_idx = other.m_transposed ? j * other.m_leadingDim + i
                                                   : i * other.m_leadingDim + j;
                } else {
                    other_idx = other.m_transposed ? i * other.m_leadingDim + j
                                                   : j * other.m_leadingDim + i;
                }

                // Compare values based on type
                switch (m_data.type) {
                    case MatrixType::u4: {
                        // 4-bit types require special handling (2 values per
                        // byte)
                        uint8_t this_byte = m_data.u4_data.data[this_idx / 2];
                        uint8_t other_byte =
                            other.m_data.u4_data.data[other_idx / 2];
                        uint8_t this_val  = (this_idx % 2 == 0)
                                                ? (this_byte & 0x0F)
                                                : (this_byte >> 4);
                        uint8_t other_val = (other_idx % 2 == 0)
                                                ? (other_byte & 0x0F)
                                                : (other_byte >> 4);
                        if (this_val != other_val)
                            return false;
                        break;
                    }
                    case MatrixType::s4: {
                        // 4-bit types require special handling (2 values per
                        // byte)
                        int8_t this_byte = m_data.s4_data.data[this_idx / 2];
                        int8_t other_byte =
                            other.m_data.s4_data.data[other_idx / 2];
                        int8_t this_val  = (this_idx % 2 == 0)
                                               ? (this_byte & 0x0F)
                                               : (this_byte >> 4);
                        int8_t other_val = (other_idx % 2 == 0)
                                               ? (other_byte & 0x0F)
                                               : (other_byte >> 4);
                        // Convert to signed
                        this_val  = (this_val & 0x08) ? (this_val | 0xF0)
                                                      : this_val;
                        other_val = (other_val & 0x08) ? (other_val | 0xF0)
                                                       : other_val;
                        if (this_val != other_val)
                            return false;
                        break;
                    }
                    case MatrixType::u8:
                        if (m_data.u8_data.data[this_idx]
                            != other.m_data.u8_data.data[other_idx])
                            return false;
                        break;
                    case MatrixType::s8:
                        if (m_data.s8_data.data[this_idx]
                            != other.m_data.s8_data.data[other_idx])
                            return false;
                        break;
                    case MatrixType::u16:
                        if (m_data.u16_data.data[this_idx]
                            != other.m_data.u16_data.data[other_idx])
                            return false;
                        break;
                    case MatrixType::s16:
                        if (m_data.s16_data.data[this_idx]
                            != other.m_data.s16_data.data[other_idx])
                            return false;
                        break;
                    case MatrixType::u32:
                        if (m_data.u32_data.data[this_idx]
                            != other.m_data.u32_data.data[other_idx])
                            return false;
                        break;
                    case MatrixType::s32:
                        if (m_data.s32_data.data[this_idx]
                            != other.m_data.s32_data.data[other_idx])
                            return false;
                        break;
                    case MatrixType::f32: {
                        float this_val  = m_data.f32_data.data[this_idx];
                        float other_val = other.m_data.f32_data.data[other_idx];

                        // FIXME: Extract this out and move tolerance to YAML
                        // parser. Use relative tolerance for floating point
                        // comparison
                        const float epsilon  = 1e-6f;
                        float       abs_diff = std::abs(this_val - other_val);
                        float       rel_tol =
                            epsilon
                            * std::max(std::abs(this_val), std::abs(other_val));
                        float tolerance = std::max(epsilon, rel_tol);

                        if (abs_diff > tolerance) {
                            return false;
                        }
                        break;
                    }
                    case MatrixType::bf16:
                        if (m_data.bf16_data.data[this_idx]
                            != other.m_data.bf16_data.data[other_idx])
                            return false;
                        break;
                    default:
                        throw std::runtime_error(
                            "Unsupported matrix type in comparison");
                }
            }
        }

        // All checks passed
        return true;
    }

    /**
     * @brief Fill matrix with random values from a uniform distribution
     *
     * Fills the matrix with random values appropriate for its data type.
     *
     * @param seed Optional seed for the random number generator (0 means use
     * time-based seed)
     */
    void Matrix::fillRandom(unsigned int seed)
    {
        // Setup random number generator
        std::mt19937 gen;
        if (seed == 0) {
            // Use time-based seed if none provided
            seed = static_cast<unsigned int>(
                std::chrono::system_clock::now().time_since_epoch().count());
        }
        gen.seed(seed);

        // Get total size using the member function
        md_t totalSize = getAllocSize();

        // Fill with random values based on type
        switch (m_data.type) {
            case MatrixType::u4: {
                // For 4-bit types, we need to pack 2 values per byte
                std::uniform_int_distribution<uint8_t> dist(
                    0, 15); // 4-bit range: 0-15
                uint8_t* data = m_data.u4_data.data;
                for (md_t i = 0; i < (totalSize + 1) / 2; i++) {
                    // Pack two 4-bit values into one byte
                    uint8_t low  = dist(gen);
                    uint8_t high = dist(gen);
                    data[i]      = (high << 4) | low;
                }
                break;
            }
            case MatrixType::s4: {
                // For 4-bit types, we need to pack 2 values per byte
                std::uniform_int_distribution<int8_t> dist(
                    -8, 7); // 4-bit signed range: -8 to 7
                int8_t* data = m_data.s4_data.data;
                for (md_t i = 0; i < (totalSize + 1) / 2; i++) {
                    // Pack two 4-bit values into one byte
                    int8_t low  = dist(gen) & 0x0F; // Ensure 4-bits
                    int8_t high = dist(gen) & 0x0F; // Ensure 4-bits
                    data[i]     = (high << 4) | low;
                }
                break;
            }
            case MatrixType::u8: {
                std::uniform_int_distribution<uint16_t> dist(
                    0, 255); // 8-bit range: 0-255
                uint8_t* data = m_data.u8_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    data[i] = static_cast<uint8_t>(dist(gen));
                }
                break;
            }
            case MatrixType::s8: {
                std::uniform_int_distribution<int16_t> dist(
                    -128, 127); // 8-bit signed range: -128 to 127
                int8_t* data = m_data.s8_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    data[i] = static_cast<int8_t>(dist(gen));
                }
                break;
            }
            case MatrixType::u16: {
                std::uniform_int_distribution<uint16_t> dist(
                    0, 65535); // 16-bit range: 0-65535
                uint16_t* data = m_data.u16_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    data[i] = dist(gen);
                }
                break;
            }
            case MatrixType::s16: {
                std::uniform_int_distribution<int16_t> dist(
                    -32768, 32767); // 16-bit signed range: -32768 to 32767
                int16_t* data = m_data.s16_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    data[i] = dist(gen);
                }
                break;
            }
            case MatrixType::u32: {
                std::uniform_int_distribution<uint32_t> dist(
                    0, 4294967295); // 32-bit range
                uint32_t* data = m_data.u32_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    data[i] = dist(gen);
                }
                break;
            }
            case MatrixType::s32: {
                std::uniform_int_distribution<int32_t> dist(
                    -2147483648, 2147483647); // 32-bit signed range
                int32_t* data = m_data.s32_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    data[i] = dist(gen);
                }
                break;
            }
            case MatrixType::f32: {
                std::uniform_real_distribution<float> dist(
                    -1.0f, 1.0f); // Float range: -1.0 to 1.0
                float* data = m_data.f32_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    data[i] = dist(gen);
                }
                break;
            }
            case MatrixType::bf16: {
                // BF16 is a 16-bit floating point format with 1 sign bit, 8
                // exponent bits, and 7 mantissa bits We'll generate random
                // floats and convert them to BF16 format
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                uint16_t* data = m_data.bf16_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    float f = dist(gen);
                    // Convert float to BF16 representation
                    // Extract sign and exponent bits
                    uint32_t floatBits;
                    std::memcpy(&floatBits, &f, sizeof(float));
                    // BF16 takes the sign bit and exponent (8 bits) and the
                    // first 7 bits of mantissa
                    uint16_t bf16 =
                        static_cast<uint16_t>((floatBits >> 16) & 0xFFFF);
                    data[i] = bf16;
                }
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported matrix type in fillRandom");
        }
    }

}} // namespace dlp::testing
