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
#include <any>
#include <chrono>   // For time-based seeding
#include <cmath>    // For std::abs
#include <cstring>  // For std::memcpy
#include <iostream> // For std::cout
#include <random>   // For random number generation

namespace dlp { namespace testing {

    /**
     * @brief Default constructor implementation
     *
     * Creates an empty matrix with default values. No memory is allocated.
     */
    Matrix::Matrix()
        : m_rows(0)
        , m_cols(0)
        , m_k(std::numeric_limits<md_t>::max())
        , m_layout(MatrixLayout::ROW_MAJOR)
        , m_leadingDim(0)
        , m_transposed(false)
        , m_reordered(false)
        , m_allocSize(0)
    {
        m_data.type = MatrixType::f32; // Default type
        // Initialize all data pointers to nullptr to prevent double-free issues
        // Since it's a union, we need to explicitly set all members to nullptr
        m_data.u4_data.data   = nullptr;
        m_data.u8_data.data   = nullptr;
        m_data.u16_data.data  = nullptr;
        m_data.u32_data.data  = nullptr;
        m_data.s4_data.data   = nullptr;
        m_data.s8_data.data   = nullptr;
        m_data.s16_data.data  = nullptr;
        m_data.s32_data.data  = nullptr;
        m_data.f32_data.data  = nullptr;
        m_data.bf16_data.data = nullptr;
    }

    /**
     * @brief Calculate the number of bytes needed for allocation based on
     * matrix type and element count
     *
     * @param type The matrix data type
     * @param elementCount The number of elements to allocate
     * @return md_t The number of bytes needed
     */
    static md_t calculateBytesForType(MatrixType type, md_t elementCount)
    {
        switch (type) {
            case MatrixType::u4:
            case MatrixType::s4:
                // 4-bit types: 2 elements per byte
                return (elementCount + 1) / 2;
            case MatrixType::u8:
            case MatrixType::s8:
                return elementCount;
            case MatrixType::u16:
            case MatrixType::s16:
            case MatrixType::bf16:
                return elementCount * sizeof(uint16_t);
            case MatrixType::u32:
            case MatrixType::s32:
                return elementCount * sizeof(uint32_t);
            case MatrixType::f32:
                return elementCount * sizeof(float);
            default:
                throw std::runtime_error(
                    "Invalid matrix type for byte calculation");
        }
    }

    /**
     * @brief Calculate automatic allocation size in bytes based on matrix
     * dimensions and type
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param layout Memory layout
     * @param leadingDim Leading dimension
     * @param type Matrix data type
     * @return md_t The number of bytes needed for allocation
     */
    static md_t calculateAutomaticAllocationBytes(md_t         rows,
                                                  md_t         cols,
                                                  MatrixLayout layout,
                                                  md_t         leadingDim,
                                                  MatrixType   type)
    {
        // Calculate element count based on layout
        md_t elementCount;
        if (layout == MatrixLayout::ROW_MAJOR) {
            elementCount = rows * leadingDim;
        } else {
            elementCount = cols * leadingDim;
        }

        // Convert element count to bytes based on type
        return calculateBytesForType(type, elementCount);
    }

    /**
     * @brief Handle user-provided allocation size (assumed to be in bytes)
     *
     * @param allocSize User-provided allocation size in bytes
     * @return md_t The allocation size in bytes (unchanged)
     */
    static md_t handleUserAllocationBytes(md_t allocSize)
    {
        // User provided size is already in bytes, use it exactly
        return allocSize;
    }

    /**
     * @brief Allocate memory for matrix data
     *
     * @param data Reference to the matrix data structure
     * @param type The matrix data type
     * @param byteCount The number of bytes to allocate
     */
    static void allocateMatrixMemory(MatrixData& data,
                                     MatrixType  type,
                                     md_t        byteCount)
    {
        switch (type) {
            case MatrixType::u4:
                data.u4_data.data = new uint8_t[byteCount];
                break;
            case MatrixType::u8:
                data.u8_data.data = new uint8_t[byteCount];
                break;
            case MatrixType::u16:
                data.u16_data.data =
                    reinterpret_cast<uint16_t*>(new uint8_t[byteCount]);
                break;
            case MatrixType::u32:
                data.u32_data.data =
                    reinterpret_cast<uint32_t*>(new uint8_t[byteCount]);
                break;
            case MatrixType::s4:
                data.s4_data.data =
                    reinterpret_cast<int8_t*>(new uint8_t[byteCount]);
                break;
            case MatrixType::s8:
                data.s8_data.data =
                    reinterpret_cast<int8_t*>(new uint8_t[byteCount]);
                break;
            case MatrixType::s16:
                data.s16_data.data =
                    reinterpret_cast<int16_t*>(new uint8_t[byteCount]);
                break;
            case MatrixType::s32:
                data.s32_data.data =
                    reinterpret_cast<int32_t*>(new uint8_t[byteCount]);
                break;
            case MatrixType::f32:
                data.f32_data.data =
                    reinterpret_cast<float*>(new uint8_t[byteCount]);
                break;
            case MatrixType::bf16:
                data.bf16_data.data =
                    reinterpret_cast<uint16_t*>(new uint8_t[byteCount]);
                break;
            default:
                throw std::runtime_error(
                    "Invalid matrix type for memory allocation");
        }
    }

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
     * @param is_reordered Whether the matrix is reordered
     * @param allocSize Override allocation size in BYTES (0 for automatic
     * calculation)
     */
    Matrix::Matrix(md_t         rows,
                   md_t         cols,
                   MatrixType   type,
                   MatrixLayout layout,
                   md_t         leadingDim,
                   bool         is_trans,
                   bool         is_reordered,
                   md_t         allocSize)
        : m_rows(rows)
        , m_cols(cols)
        , m_layout(layout)
        , m_transposed(is_trans)
        , m_reordered(is_reordered)
        , m_allocSize(allocSize)
    {
        // Calculate leading dimension if not specified
        if (leadingDim == 0) {
            m_leadingDim = (layout == MatrixLayout::ROW_MAJOR) ? cols : rows;
        } else {
            m_leadingDim = leadingDim;
        }

        m_data.type = type;

        // Use user-provided allocation size if specified, otherwise calculate
        // it
        md_t bytesToAllocate;
        if (allocSize > 0) {
            // User provided allocation size in bytes - use it exactly
            bytesToAllocate = handleUserAllocationBytes(allocSize);
        } else {
            // Calculate allocation size based on matrix dimensions and type
            bytesToAllocate = calculateAutomaticAllocationBytes(
                rows, cols, layout, m_leadingDim, type);
        }

        // Allocate memory using the helper function
        allocateMatrixMemory(m_data, type, bytesToAllocate);
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
     * @brief Copy constructor implementation
     *
     * Creates a deep copy of the source matrix.
     *
     * @param other The matrix to copy from
     */
    Matrix::Matrix(const Matrix& other)
        : m_rows(other.m_rows)
        , m_cols(other.m_cols)
        , m_k(other.m_k)
        , m_layout(other.m_layout)
        , m_leadingDim(other.m_leadingDim)
        , m_transposed(other.m_transposed)
        , m_reordered(other.m_reordered)
        , m_allocSize(other.m_allocSize)
    {
        m_data.type = other.m_data.type;

        // Use the stored allocation size if available, otherwise calculate it
        md_t bytesToAllocate;
        if (m_allocSize > 0) {
            // User provided allocation size in bytes - use it exactly
            bytesToAllocate = handleUserAllocationBytes(m_allocSize);
        } else {
            // Calculate allocation size based on matrix dimensions and type
            bytesToAllocate = calculateAutomaticAllocationBytes(
                m_rows, m_cols, m_layout, m_leadingDim, m_data.type);
        }

        // Allocate memory using the helper function
        allocateMatrixMemory(m_data, m_data.type, bytesToAllocate);

        // Copy the data - we need to copy the exact number of bytes allocated
        switch (m_data.type) {
            case MatrixType::u4:
            case MatrixType::s4:
            case MatrixType::u8:
            case MatrixType::s8:
                std::memcpy(m_data.u8_data.data, other.m_data.u8_data.data,
                            bytesToAllocate);
                break;
            case MatrixType::u16:
            case MatrixType::s16:
            case MatrixType::bf16:
                std::memcpy(m_data.u16_data.data, other.m_data.u16_data.data,
                            bytesToAllocate);
                break;
            case MatrixType::u32:
            case MatrixType::s32:
                std::memcpy(m_data.u32_data.data, other.m_data.u32_data.data,
                            bytesToAllocate);
                break;
            case MatrixType::f32:
                std::memcpy(m_data.f32_data.data, other.m_data.f32_data.data,
                            bytesToAllocate);
                break;
            default:
                throw std::runtime_error(
                    "Invalid matrix type in copy constructor");
        }
    }

    /**
     * @brief Destructor implementation
     *
     * Frees allocated memory based on the matrix type.
     */
    Matrix::~Matrix()
    {
        // Since all allocations are now done as uint8_t arrays, we can simplify
        // the cleanup Also check for nullptr to handle default-constructed
        // matrices
        switch (m_data.type) {
            case MatrixType::u4:
                if (m_data.u4_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.u4_data.data);
                    m_data.u4_data.data = nullptr;
                }
                break;
            case MatrixType::u8:
                if (m_data.u8_data.data != nullptr) {
                    delete[] m_data.u8_data.data;
                    m_data.u8_data.data = nullptr;
                }
                break;
            case MatrixType::u16:
                if (m_data.u16_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.u16_data.data);
                    m_data.u16_data.data = nullptr;
                }
                break;
            case MatrixType::u32:
                if (m_data.u32_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.u32_data.data);
                    m_data.u32_data.data = nullptr;
                }
                break;
            case MatrixType::s4:
                if (m_data.s4_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.s4_data.data);
                    m_data.s4_data.data = nullptr;
                }
                break;
            case MatrixType::s8:
                if (m_data.s8_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.s8_data.data);
                    m_data.s8_data.data = nullptr;
                }
                break;
            case MatrixType::s16:
                if (m_data.s16_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.s16_data.data);
                    m_data.s16_data.data = nullptr;
                }
                break;
            case MatrixType::s32:
                if (m_data.s32_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.s32_data.data);
                    m_data.s32_data.data = nullptr;
                }
                break;
            case MatrixType::f32:
                if (m_data.f32_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.f32_data.data);
                    m_data.f32_data.data = nullptr;
                }
                break;
            case MatrixType::bf16:
                if (m_data.bf16_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.bf16_data.data);
                    m_data.bf16_data.data = nullptr;
                }
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

        // Free existing resources based on current type
        switch (m_data.type) {
            case MatrixType::u4:
                if (m_data.u4_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.u4_data.data);
                    m_data.u4_data.data = nullptr;
                }
                break;
            case MatrixType::u8:
                if (m_data.u8_data.data != nullptr) {
                    delete[] m_data.u8_data.data;
                    m_data.u8_data.data = nullptr;
                }
                break;
            case MatrixType::u16:
                if (m_data.u16_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.u16_data.data);
                    m_data.u16_data.data = nullptr;
                }
                break;
            case MatrixType::u32:
                if (m_data.u32_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.u32_data.data);
                    m_data.u32_data.data = nullptr;
                }
                break;
            case MatrixType::s4:
                if (m_data.s4_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.s4_data.data);
                    m_data.s4_data.data = nullptr;
                }
                break;
            case MatrixType::s8:
                if (m_data.s8_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.s8_data.data);
                    m_data.s8_data.data = nullptr;
                }
                break;
            case MatrixType::s16:
                if (m_data.s16_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.s16_data.data);
                    m_data.s16_data.data = nullptr;
                }
                break;
            case MatrixType::s32:
                if (m_data.s32_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.s32_data.data);
                    m_data.s32_data.data = nullptr;
                }
                break;
            case MatrixType::f32:
                if (m_data.f32_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.f32_data.data);
                    m_data.f32_data.data = nullptr;
                }
                break;
            case MatrixType::bf16:
                if (m_data.bf16_data.data != nullptr) {
                    delete[] reinterpret_cast<uint8_t*>(m_data.bf16_data.data);
                    m_data.bf16_data.data = nullptr;
                }
                break;
            default:
                break;
        }

        // Copy metadata
        m_rows       = other.m_rows;
        m_cols       = other.m_cols;
        m_k          = other.m_k;
        m_layout     = other.m_layout;
        m_leadingDim = other.m_leadingDim;
        m_transposed = other.m_transposed;
        m_reordered  = other.m_reordered;
        m_allocSize  = other.m_allocSize;
        m_data.type  = other.m_data.type;

        // Use the stored allocation size if available, otherwise calculate it
        md_t bytesToAllocate;
        if (m_allocSize > 0) {
            // User provided allocation size in bytes - use it exactly
            bytesToAllocate = handleUserAllocationBytes(m_allocSize);
        } else {
            // Calculate allocation size based on matrix dimensions and type
            bytesToAllocate = calculateAutomaticAllocationBytes(
                m_rows, m_cols, m_layout, m_leadingDim, m_data.type);
        }

        // Allocate memory using the helper function
        allocateMatrixMemory(m_data, m_data.type, bytesToAllocate);

        // Copy the data - we need to copy the exact number of bytes allocated
        switch (m_data.type) {
            case MatrixType::u4:
            case MatrixType::s4:
            case MatrixType::u8:
            case MatrixType::s8:
                std::memcpy(m_data.u8_data.data, other.m_data.u8_data.data,
                            bytesToAllocate);
                break;
            case MatrixType::u16:
            case MatrixType::s16:
            case MatrixType::bf16:
                std::memcpy(m_data.u16_data.data, other.m_data.u16_data.data,
                            bytesToAllocate);
                break;
            case MatrixType::u32:
            case MatrixType::s32:
                std::memcpy(m_data.u32_data.data, other.m_data.u32_data.data,
                            bytesToAllocate);
                break;
            case MatrixType::f32:
                std::memcpy(m_data.f32_data.data, other.m_data.f32_data.data,
                            bytesToAllocate);
                break;
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
     * Returns the allocation size in elements. If a custom allocation
     * size was provided during construction (in bytes), this function
     * converts it back to element count for compatibility.
     *
     * @return md_t The allocation size in number of elements
     */
    md_t Matrix::getAllocSize() const
    {
        // If user provided a custom allocation size (in bytes), convert back to
        // elements
        if (m_allocSize > 0) {
            switch (m_data.type) {
                case MatrixType::u4:
                case MatrixType::s4:
                    // 4-bit types: 2 elements per byte
                    return m_allocSize * 2;
                case MatrixType::u8:
                case MatrixType::s8:
                    return m_allocSize;
                case MatrixType::u16:
                case MatrixType::s16:
                case MatrixType::bf16:
                    return m_allocSize / sizeof(uint16_t);
                case MatrixType::u32:
                case MatrixType::s32:
                    return m_allocSize / sizeof(uint32_t);
                case MatrixType::f32:
                    return m_allocSize / sizeof(float);
                default:
                    throw std::runtime_error(
                        "Invalid matrix type in getAllocSize");
            }
        }

        // Otherwise calculate based on dimensions and layout (returns element
        // count)
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
     * @brief Set the k dimension for tolerance calculation
     *
     * @param k The k dimension for tolerance calculation
     */
    void Matrix::setK(md_t k)
    {
        m_k = k;
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
                        // Formula : 2*k*1.2e-6;

                        float epsilon;
                        if (m_k == std::numeric_limits<md_t>::max()) {
                            epsilon = 2 * m_cols * 1.2e-6f; // Default to column
                                                            // size
                        } else {
                            epsilon = 2 * m_k * 1.2e-6f; // Use user-defined k
                        }
                        float abs_diff = std::abs(this_val - other_val);
                        float rel_tol =
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
     * @brief Compare two matrices for inequality
     *
     * Checks if two matrices differ in dimensions, type, or content.
     *
     * @param other The matrix to compare with
     * @return bool True if matrices are not equal, false otherwise
     */
    bool Matrix::operator!=(const Matrix& other) const
    {
        return !(*this == other);
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
#if 1
                std::uniform_real_distribution<float> dist(
                    -1.0f, 1.0f); // Float range: -1.0 to 1.0
                float* data = m_data.f32_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    data[i] = dist(gen);
                }
                break;
#else
                // We are going to use int values then convert them to float
                std::uniform_int_distribution<int32_t> dist(
                    -10, +10); // 32-bit range
                float* data = m_data.f32_data.data;
                for (md_t i = 0; i < totalSize; i++) {
                    data[i] = dist(gen);
                }
#endif
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

    /**
     * @brief Fill matrix with a single value
     *
     * Fills the matrix with a single value of the appropriate data type.
     *
     * @param value The value to fill the matrix with
     */
    void Matrix::fillValue(std::any value)
    {
        switch (m_data.type) {
            case MatrixType::f32:
                for (md_t i = 0; i < getAllocSize(); i++) {
                    m_data.f32_data.data[i] = std::any_cast<float>(value);
                }
                break;
            default:
                throw std::runtime_error(
                    "Unsupported matrix type in fillValue");
        }
    }

}} // namespace dlp::testing
