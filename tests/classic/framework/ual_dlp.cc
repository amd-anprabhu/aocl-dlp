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
 * @file ual_dlp.cc
 * @brief Implementation of the DLP Unified Abstraction Layer
 *
 * This file contains the implementation of the DLP-based UAL, providing
 * optimized matrix operations with support for various data types, memory
 * layouts, and virtual transposition.
 */

#include "framework/ual_dlp.hh"
#include <iostream>

extern "C"
{
#include "aocl_dlp.h"
}

namespace dlp::testing::classic {

/**
 * @brief Constructor for UalDlp
 *
 * Initializes a DLP-based Unified Abstraction Layer implementation.
 */
UalDlp::UalDlp()
    : IUal(UALType::DLP)
{
}

/**
 * @brief Get the UAL implementation type
 *
 * @return UALType::DLP for this implementation
 */
UALType
UalDlp::getUALType() const
{
    return UALType::DLP;
}

/**
 * @brief Convert UAL type to human-readable string
 *
 * @param type The UAL type to convert
 * @return std::string Human-readable description
 */
std::string
UalDlp::toString(UALType type)
{
    switch (type) {
        case UALType::DLP:
            return "Deep Learning Primitives";
        case UALType::MKL:
            return "Intel MKL";
        case UALType::ONEDNN:
            return "OneDNN";
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
UalDlp::reorder(const Matrix& in, Matrix& out, MatrixType accType)
{
    // Use effective (logical) dimensions for reordering
    md_t effective_rows = in.getEffectiveRows();
    md_t effective_cols = in.getEffectiveCols();

    md_t alloc_size = aocl_get_reorder_buf_size_f32f32f32of32(
        in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
        in.isTransposed() ? 't' : 'n', 'B', effective_cols, effective_rows);

    // Convert element count to bytes for f32 type
    md_t alloc_bytes = alloc_size * sizeof(float);

    // Create output matrix with correct parameter order:
    // Matrix(rows, cols, type, layout, leadingDim, transposed, reordered,
    // allocSize)
    out =
        Matrix(in.getRows(), in.getCols(), in.getMatrixType(), in.getLayout(),
               in.getLeadingDimension(), in.isTransposed(), true, alloc_bytes);

    char layout = in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';
    switch (in.getMatrixType()) {
        case MatrixType::f32:
            aocl_reorder_f32f32f32of32(
                layout, in.isTransposed() ? 't' : 'n', 'B',
                reinterpret_cast<const float*>(
                    in.getMatrixData().getMatrixPtr()),
                reinterpret_cast<float*>(out.getMatrixData().getMatrixPtr()),
                effective_rows, effective_cols, in.getLeadingDimension());
            break;
        default:
            return false;
    }
    return true;
}

/**
 * @brief Validate GEMM parameters for correctness
 *
 * NOTE: This client-side validation is not ideal - proper error handling
 * should be implemented at the library level to provide consistent
 * parameter validation across all UAL implementations.
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix
 * @return bool True if parameters are valid, false otherwise
 */
bool
UalDlp::checkValidGemmParams(const Matrix& A, const Matrix& B, const Matrix& C)
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
UalDlp::gemm(const Matrix& A, const Matrix& B, Matrix& C, MatrixType accType)
{
    // Validate parameters first
    // NOTE: This client-side validation is not ideal - proper error handling
    // should be implemented at the library level to provide consistent
    // parameter validation across all UAL implementations.
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

    // Validate leading dimensions
    if (C.getLayout() == MatrixLayout::ROW_MAJOR) {
        if (C.getLeadingDimension() < C.getCols()) {
            std::cerr << "ERROR: Invalid leading dimension for matrix C. "
                      << "For row-major layout, ldc ("
                      << C.getLeadingDimension()
                      << ") must be >= number of columns (" << C.getCols()
                      << ")" << std::endl;
            return false;
        }
    } else {
        if (C.getLeadingDimension() < C.getRows()) {
            std::cerr << "ERROR: Invalid leading dimension for matrix C. "
                      << "For column-major layout, ldc ("
                      << C.getLeadingDimension()
                      << ") must be >= number of rows (" << C.getRows() << ")"
                      << std::endl;
            return false;
        }
    }

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>():

            aocl_gemm_f32f32f32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), 1.0,
                reinterpret_cast<float*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), isAReordered,
                reinterpret_cast<float*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), isBReordered, 1.0,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;

        default:
            return false;
    }
}

} // namespace dlp::testing::classic
