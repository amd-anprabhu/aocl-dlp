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

#include "aocl_dlp.h"
#include "framework/ual.hh"
#include "framework/ual_factory.hh"
#include "test_config.hh"
#include "utils/yaml_parser.hh"
#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace dlp::testing;
using dlp::testing::IUal;
using dlp::testing::UalFactory;

// Configuration structure for YAML-driven tests
struct GemmTestConfig
{
    std::string  name;
    MatrixType   a_type, b_type, c_type, acc_type;
    MatrixLayout storage_format;
    md_t         m, n, k;
    md_t         lda, ldb, ldc;
    double       alpha, beta; // Changed from float to double for consistency
    bool         transA, transB;
    bool         reorderA, reorderB;
    bool         packA, packB; // TODO: Pack parameters not yet used in tests

    // Default constructor (required by GoogleTest)
    GemmTestConfig()
        : name("default")
        , m(1)
        , n(1)
        , k(1)
        , lda(1)
        , ldb(1)
        , ldc(1)
        , alpha(1.0)
        , beta(0.0)
        , transA(false)
        , transB(false)
        , reorderA(false)
        , reorderB(false)
        , packA(false)
        , packB(false)
    {
    }

    // Constructor for easy initialization
    GemmTestConfig(const std::string& test_name,
                   MatrixType         a_type_,
                   MatrixType         b_type_,
                   MatrixType         c_type_,
                   MatrixType         acc_type_,
                   MatrixLayout       storage_format_,
                   md_t               m_,
                   md_t               n_,
                   md_t               k_,
                   md_t               lda_,
                   md_t               ldb_,
                   md_t               ldc_,
                   double             alpha_,
                   double             beta_,
                   bool               transA_,
                   bool               transB_,
                   bool               reorderA_,
                   bool               reorderB_,
                   bool               packA_,
                   bool               packB_)
        : name(test_name)
        , a_type(a_type_)
        , b_type(b_type_)
        , c_type(c_type_)
        , acc_type(acc_type_)
        , storage_format(storage_format_)
        , m(m_)
        , n(n_)
        , k(k_)
        , lda(lda_)
        , ldb(ldb_)
        , ldc(ldc_)
        , alpha(alpha_)
        , beta(beta_)
        , transA(transA_)
        , transB(transB_)
        , reorderA(reorderA_)
        , reorderB(reorderB_)
        , packA(packA_)
        , packB(packB_)
    {
    }

    // Equality operator for Google Test
    bool operator==(const GemmTestConfig& other) const
    {
        return name == other.name && m == other.m && n == other.n
               && k == other.k && lda == other.lda && ldb == other.ldb
               && ldc == other.ldc && alpha == other.alpha && beta == other.beta
               && transA == other.transA && transB == other.transB
               && reorderA == other.reorderA && reorderB == other.reorderB
               && packA == other.packA && packB == other.packB;
    }
};

// Custom printer for better test output
void
PrintTo(const GemmTestConfig& config, std::ostream* os)
{
    *os << config.name << "_(" << config.m << "x" << config.n << "x"
        << config.k;
    if (config.transA || config.transB) {
        *os << "_trans" << (config.transA ? "A" : "")
            << (config.transB ? "B" : "");
    }
    if (config.storage_format == MatrixLayout::COLUMN_MAJOR) {
        *os << "_colMajor";
    } else {
        *os << "_rowMajor";
    }
    if (config.reorderA || config.reorderB) {
        *os << "_reorder" << (config.reorderA ? "A" : "")
            << (config.reorderB ? "B" : "");
    }
    *os << ")";
}

// Helper function to generate meaningful test names
std::string
generateTestName(const MicroTest& microTest,
                 size_t           testSetIndex,
                 size_t           configIndex)
{
    std::ostringstream name;
    name << "yaml_" << testSetIndex << "_M" << microTest.getM() << "_N"
         << microTest.getN() << "_K" << microTest.getK();
    if (microTest.getTransA())
        name << "_transA";
    if (microTest.getTransB())
        name << "_transB";
    if (microTest.getReorderA())
        name << "_reorderA";
    if (microTest.getReorderB())
        name << "_reorderB";
    name << "_" << configIndex;
    return name.str();
}

// Function to load test configurations from YAML
std::vector<GemmTestConfig>
loadTestConfigurations(const std::string& yaml_file)
{
    std::vector<GemmTestConfig> configs;
    const size_t                MAX_CONFIGS_PER_TEST_SET =
        50000; // Reasonable limit to prevent millions of tests

    try {
        YamlParser parser(yaml_file, "gemm_tests");
        parser.setYieldType(YieldType::CARTESIAN_PRODUCT);

        size_t microTestCount = parser.getMicroTestCount();

        for (size_t i = 0; i < microTestCount; ++i) {
            // Get a mutable reference to the MicroTest for iteration
            // Note: This design could be improved by making MicroTest iteration
            // const-correct
            MicroTest& microTest =
                const_cast<MicroTest&>(parser.getMicroTest());
            size_t total_combinations = microTest.getSize();
            size_t test_count =
                std::min(total_combinations, MAX_CONFIGS_PER_TEST_SET);

            std::cout << "Test set " << i << ": Using " << test_count
                      << " out of " << total_combinations
                      << " total combinations" << std::endl;

            for (size_t j = 0; j < test_count; ++j) {
                // Generate meaningful test name
                std::string testName = generateTestName(microTest, i, j);

                // Extract configuration from parser
                configs.emplace_back(
                    testName,                     // name
                    microTest.getAType(),         // a_type
                    microTest.getBType(),         // b_type
                    microTest.getCType(),         // c_type
                    microTest.getAccType(),       // acc_type
                    microTest.getStorageFormat(), // storage_format
                    microTest.getM(),             // m
                    microTest.getN(),             // n
                    microTest.getK(),             // k
                    microTest.getLDA(),           // lda
                    microTest.getLDB(),           // ldb
                    microTest.getLDC(),           // ldc
                    microTest.getAlpha(),         // alpha
                    microTest.getBeta(),          // beta
                    microTest.getTransA(),        // transA
                    microTest.getTransB(),        // transB
                    microTest.getReorderA(),      // reorderA
                    microTest.getReorderB(),      // reorderB
                    microTest.getPackA(),         // packA
                    microTest.getPackB()          // packB
                );
                if (j < test_count - 1) {
                    microTest.next();
                }
            }
            // Move to next test configuration
            if (i < microTestCount - 1) {
                parser.next();
            }
        }

        std::cout << "Loaded " << configs.size() << " test configurations from "
                  << yaml_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading YAML configuration: " << e.what()
                  << std::endl;
        // Use Google Test's failure mechanism instead of silent failure
        ADD_FAILURE() << "Failed to load YAML configuration: " << e.what();
        return configs; // Return empty vector but with explicit test failure
    }

    return configs;
}

// Function to create additional test configurations programmatically
std::vector<GemmTestConfig>
createAdditionalTestConfigs()
{
    std::vector<GemmTestConfig> configs;

    // Add some programmatically generated test cases
    configs.emplace_back("small_square",          // name
                         MatrixType::f32,         // a_type
                         MatrixType::f32,         // b_type
                         MatrixType::f32,         // c_type
                         MatrixType::f32,         // acc_type
                         MatrixLayout::ROW_MAJOR, // storage_format
                         16, 16, 16,              // m, n, k
                         16, 16, 16,              // lda, ldb, ldc
                         1.0, 0.0,                // alpha, beta
                         false, false,            // transA, transB
                         false, false,            // reorderA, reorderB
                         false, false             // packA, packB
    );

    configs.emplace_back("rectangular_1",         // name
                         MatrixType::f32,         // a_type
                         MatrixType::f32,         // b_type
                         MatrixType::f32,         // c_type
                         MatrixType::f32,         // acc_type
                         MatrixLayout::ROW_MAJOR, // storage_format
                         32, 16, 24,              // m, n, k
                         24, 16, 16,              // lda, ldb, ldc
                         1.5, 0.5,                // alpha, beta
                         false, false,            // transA, transB
                         false, false,            // reorderA, reorderB
                         false, false             // packA, packB
    );

    configs.emplace_back("transposed_A",          // name
                         MatrixType::f32,         // a_type
                         MatrixType::f32,         // b_type
                         MatrixType::f32,         // c_type
                         MatrixType::f32,         // acc_type
                         MatrixLayout::ROW_MAJOR, // storage_format
                         20, 20, 20,              // m, n, k
                         20, 20, 20,              // lda, ldb, ldc
                         2.0, 1.0,                // alpha, beta
                         false, false,            // transA, transB
                         false, false,            // reorderA, reorderB
                         false, false             // packA, packB
    );

    configs.emplace_back("edge_case_1x1",         // name
                         MatrixType::f32,         // a_type
                         MatrixType::f32,         // b_type
                         MatrixType::f32,         // c_type
                         MatrixType::f32,         // acc_type
                         MatrixLayout::ROW_MAJOR, // storage_format
                         1, 1, 1,                 // m, n, k
                         1, 1, 1,                 // lda, ldb, ldc
                         1.0, 0.0,                // alpha, beta
                         false, false,            // transA, transB
                         false, false,            // reorderA, reorderB
                         false, false             // packA, packB
    );

    return configs;
}

// Static initialization of test configurations
// This runs before main() and loads all configurations for parameterized tests
static std::vector<GemmTestConfig>
initializeTestConfigurations()
{
    std::vector<GemmTestConfig> all_configs;

    // Load YAML configurations
    std::string config_file  = TEST_CONFIG_DIR "/gemm_test_config.yaml";
    auto        yaml_configs = loadTestConfigurations(config_file);

    // Load programmatic configurations
    auto prog_configs = createAdditionalTestConfigs();

    // Combine all configurations
    all_configs.reserve(yaml_configs.size() + prog_configs.size());
    all_configs.insert(all_configs.end(), yaml_configs.begin(),
                       yaml_configs.end());
    all_configs.insert(all_configs.end(), prog_configs.begin(),
                       prog_configs.end());

    std::cout << "Total test configurations initialized: " << all_configs.size()
              << std::endl;

    return all_configs;
}

// Global configurations - initialized once at startup
static const std::vector<GemmTestConfig> g_all_test_configs =
    initializeTestConfigurations();

// ============================================================================
// PARAMETERIZED TEST FIXTURE
// ============================================================================

class GemmParameterizedTest : public ::testing::TestWithParam<GemmTestConfig>
{
  protected:
    void SetUp() override
    {
        config_ = GetParam();
        std::cout << "Setting up test: " << config_.name << " (" << config_.m
                  << "x" << config_.n << "x" << config_.k << ")" << std::endl;
    }

    void TearDown() override
    {
        // Optional cleanup per test
    }

    // Helper method to run the actual GEMM test
    void RunGemmTest()
    {
        // Create matrices based on configuration
        MatrixLayout layout = config_.storage_format;

        // Determine effective dimensions considering transposition
        md_t a_rows = config_.transA ? config_.k : config_.m;
        md_t a_cols = config_.transA ? config_.m : config_.k;
        md_t b_rows = config_.transB ? config_.n : config_.k;
        md_t b_cols = config_.transB ? config_.k : config_.n;

        Matrix A(a_rows, a_cols, config_.a_type, layout, config_.lda,
                 config_.transA);
        Matrix B(b_rows, b_cols, config_.b_type, layout, config_.ldb,
                 config_.transB);
        Matrix C(config_.m, config_.n, config_.c_type, layout, config_.ldc,
                 false);
        Matrix C_ref(config_.m, config_.n, config_.acc_type, layout,
                     config_.ldc, false);

        // Initialize matrices with deterministic random values
        A.fillRandom(42 + config_.m); // Use configuration to vary seed
        B.fillRandom(43 + config_.n);
        C.fillRandom(44 + config_.k);

        // Make a copy of C for reference computation
        C_ref = C;

        // Apply reordering if specified
        if (config_.reorderA) {
            A.setReordered(true);
        }
        if (config_.reorderB) {
            B.setReordered(true);
        }

        // TODO: The current UAL interface doesn't support alpha/beta parameters
        // Future enhancement: Add alpha/beta support to the GEMM interface
        // For now, alpha/beta values are stored in config but not used in
        // computation

        // Perform GEMM with DLP implementation
        std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
        ASSERT_TRUE(ual_dlp != nullptr) << "Failed to create DLP UAL";

        bool dlp_result = ual_dlp->gemm(A, B, C, config_.acc_type);
        EXPECT_TRUE(dlp_result) << "DLP GEMM failed for test: " << config_.name;

        // Perform GEMM with reference implementation
        std::unique_ptr<IUal> ual_ref = UalFactory::createUal(UALType::REF);
        ASSERT_TRUE(ual_ref != nullptr) << "Failed to create REF UAL";

        bool ref_result = ual_ref->gemm(A, B, C_ref, config_.acc_type);
        EXPECT_TRUE(ref_result)
            << "Reference GEMM failed for test: " << config_.name;

        // Compare results
        EXPECT_EQ(C, C_ref)
            << "DLP and Reference results differ for test: " << config_.name
            << " (M=" << config_.m << ", N=" << config_.n << ", K=" << config_.k
            << ")";
    }

  protected:
    GemmTestConfig config_;
};

// The actual parameterized test - each configuration becomes a separate test
// case
TEST_P(GemmParameterizedTest, CompareImplementations)
{
    RunGemmTest();
}

// Register all configurations as individual test cases
INSTANTIATE_TEST_SUITE_P(
    YamlConfigurations,
    GemmParameterizedTest,
    ::testing::ValuesIn(g_all_test_configs),
    [](const ::testing::TestParamInfo<GemmTestConfig>& info) {
        return info.param.name;
    });

// ============================================================================
// ADDITIONAL NON-PARAMETERIZED TESTS
// ============================================================================

// Original basic test - kept as an example
TEST(GEMMTest, Basic)
{
    // Test setup
    md_t m = 2;
    md_t n = 3;
    md_t k = 4;

    // Create matrices
    Matrix A(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C_ref(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    // Initialize matrices with random values
    A.fillRandom(42); // Using fixed seed for reproducibility
    B.fillRandom(43);
    C.fillRandom(44);
    // Make a copy of C for reference
    C_ref = C;

    // Perform GEMM
    std::unique_ptr<IUal> ual = UalFactory::createUal(UALType::DLP);
    ual->gemm(A, B, C, MatrixType::f32);

    // Perform GEMM with reference implementation
    ual = UalFactory::createUal(UALType::REF);
    ual->gemm(A, B, C_ref, MatrixType::f32);

    EXPECT_EQ(C, C_ref);
}

// Separate test for matrix creation edge cases
TEST(GEMMTest, EdgeCases)
{
    // Test 1x1 matrices - use explicit constructor with layout parameter to
    // avoid ambiguity
    Matrix A_small(1u, 1u, MatrixType::f32, MatrixLayout::ROW_MAJOR);
    Matrix B_small(1u, 1u, MatrixType::f32, MatrixLayout::ROW_MAJOR);
    Matrix C_small(1u, 1u, MatrixType::f32, MatrixLayout::ROW_MAJOR);
    Matrix C_small_ref(1u, 1u, MatrixType::f32, MatrixLayout::ROW_MAJOR);

    A_small.fillRandom(1);
    B_small.fillRandom(2);
    C_small.fillRandom(3);
    C_small_ref = C_small;

    std::unique_ptr<IUal> ual = UalFactory::createUal(UALType::DLP);
    ASSERT_TRUE(ual->gemm(A_small, B_small, C_small, MatrixType::f32));

    ual = UalFactory::createUal(UALType::REF);
    ASSERT_TRUE(ual->gemm(A_small, B_small, C_small_ref, MatrixType::f32));

    EXPECT_EQ(C_small, C_small_ref);
}
