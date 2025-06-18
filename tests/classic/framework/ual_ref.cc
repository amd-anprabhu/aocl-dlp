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
 * @file ual_ref.cc
 * @brief Implementation of the Reference Unified Abstraction Layer
 *
 * This file contains the implementation of the reference-based UAL, providing
 * basic matrix operations with support for various data types, memory
 * layouts, and virtual transposition.
 */

#include "framework/ual_ref.hh"
#include "ref/gemm_ref.hh"

extern "C"
{
#include "aocl_dlp.h"
}

namespace dlp::testing::classic {

/**
 * @brief Constructor for UalRef
 *
 * Initializes a reference-based Unified Abstraction Layer implementation.
 */
UalRef::UalRef()
    : IUal(UALType::REF)
{
}

/**
 * @brief Get the UAL implementation type
 *
 * @return UALType::REF for this implementation
 */
UALType
UalRef::getUALType() const
{
    return UALType::REF;
}

/**
 * @brief Convert UAL type to human-readable string
 *
 * @param type The UAL type to convert
 * @return std::string Human-readable description
 */
std::string
UalRef::toString(UALType type)
{
    switch (type) {
        case UALType::DLP:
            return "Deep Learning Primitives";
        case UALType::MKL:
            return "Intel MKL";
        case UALType::ONEDNN:
            return "OneDNN";
        case UALType::REF:
            return "Reference Implementation";
        default:
            return "Unknown UAL";
    }
}

/**
 * @brief Public reorder interface that unpacks Matrix object
 *
 * @param in Input matrix to reorder
 * @param out Output matrix to store reordered data
 * @param accType Target accumulation type
 * @return bool Success status
 */
bool
UalRef::reorder(const Matrix& in, Matrix& out, MatrixType accType)
{
    // Reference implementation doesn't actually reorder - just copy the input
    // to output
    out = in;
    return true;
}

/**
 * @brief Internal implementation of reorder with direct parameter access
 *
 * @param A Pointer to matrix data
 * @param AType Type of matrix data
 * @param rows Number of rows
 * @param cols Number of columns
 * @param leadingDim Leading dimension
 * @param layout Memory layout
 * @param transposed Whether the matrix is transposed
 * @param accType Target accumulation type
 * @return bool Success status
 */
bool
UalRef::reorder(void*        A,
                MatrixType   AType,
                uint32_t     rows,
                uint32_t     cols,
                uint32_t     leadingDim,
                MatrixLayout layout,
                bool         transposed,
                MatrixType   accType)
{
    // Implementation would depend on reference library calls
    // This is a placeholder
    return true;
}

/**
 * @brief Validate GEMM parameters for correctness
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix
 * @return bool True if parameters are valid, false otherwise
 */
bool
UalRef::checkValidGemmParams(const Matrix& A, const Matrix& B, const Matrix& C)
{
    // Get effective dimensions considering transposition
    uint32_t m = A.getEffectiveRows(); // Rows of A (and C)
    uint32_t n = B.getEffectiveCols(); // Cols of B (and C)
    uint32_t k = A.getEffectiveCols(); // Cols of A, Rows of B

    // Check basic dimensions - must be positive
    if (m <= 0 || n <= 0 || k <= 0) {
        return false;
    }

    // Check dimension compatibility for matrix multiplication
    if (A.getEffectiveCols() != B.getEffectiveRows()) {
        return false;
    }

    // Check that C has correct dimensions
    if (C.getEffectiveRows() != m || C.getEffectiveCols() != n) {
        return false;
    }

    bool row_stored = (A.getLayout() == MatrixLayout::ROW_MAJOR);
    bool col_stored = (A.getLayout() == MatrixLayout::COLUMN_MAJOR);

    // All matrices should have the same layout
    if (A.getLayout() != B.getLayout() || A.getLayout() != C.getLayout()) {
        return false;
    }

    // Check leading dimension for matrix A
    if (row_stored) {
        // Row-major storage
        if ((!A.isTransposed()
             && A.getLeadingDimension() < A.getEffectiveCols())
            || (A.isTransposed()
                && A.getLeadingDimension() < A.getEffectiveRows())) {
            return false;
        }
    } else if (col_stored) {
        // Column-major storage
        if ((!A.isTransposed()
             && A.getLeadingDimension() < A.getEffectiveRows())
            || (A.isTransposed()
                && A.getLeadingDimension() < A.getEffectiveCols())) {
            return false;
        }
    }

    // Check leading dimension for matrix B
    if (row_stored) {
        // Row-major storage
        if ((!B.isTransposed()
             && B.getLeadingDimension() < B.getEffectiveCols())
            || (B.isTransposed()
                && B.getLeadingDimension() < B.getEffectiveRows())) {
            return false;
        }
    } else if (col_stored) {
        // Column-major storage
        if ((!B.isTransposed()
             && B.getLeadingDimension() < B.getEffectiveRows())
            || (B.isTransposed()
                && B.getLeadingDimension() < B.getEffectiveCols())) {
            return false;
        }
    }

    // Check leading dimension for matrix C
    if (row_stored) {
        // Row-major storage: C is always m x n, so ldc >= n
        if (C.getLeadingDimension() < C.getEffectiveCols()) {
            return false;
        }
    } else if (col_stored) {
        // Column-major storage: C is always m x n, so ldc >= m
        if (C.getLeadingDimension() < C.getEffectiveRows()) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Public GEMM interface that unpacks Matrix objects
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix
 * @param accType Accumulation type
 * @return bool Success status
 */
bool
UalRef::gemm(const Matrix& A, const Matrix& B, Matrix& C, MatrixType accType)
{
    // Validate parameters first
    if (!checkValidGemmParams(A, B, C)) {
        return false;
    }

    uint64_t type = encode_types(A.getMatrixType(), B.getMatrixType(),
                                 C.getMatrixType(), accType);

    char transA = A.isTransposed() ? 't' : 'n';
    char transB = B.isTransposed() ? 't' : 'n';
    // char transC = C.isTransposed() ? 't' : 'n';

    char layoutA = A.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';
    // char layoutB = B.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';
    // char layoutC = C.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    // Might have 'p' for packed
    char isAReordered = A.isReordered() ? 'r' : 'n';
    char isBReordered = B.isReordered() ? 'r' : 'n';

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>():

            dlp::testing::classic::ref::aocl_gemm_f32f32f32of32_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), 1.0,
                reinterpret_cast<const float*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const float*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), 1.0,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;

        default:
            return false;
    }
}
} // namespace dlp::testing::classic
