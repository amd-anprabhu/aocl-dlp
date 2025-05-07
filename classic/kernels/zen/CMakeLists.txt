#
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (
# INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# CMake file for Zen kernels
# Get all source files in the zen directory recursively
dlp_glob_sources_recursive(ZEN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR})

# Create the zen kernel object library
add_library(dlp_kernel_zen OBJECT ${ZEN_SOURCES})

# Set include directories for zen kernels
set(ZEN_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/f32f32f32
)

# Make all include directories public
target_include_directories(dlp_kernel_zen
    PRIVATE
        ${ZEN_INCLUDES}
        ${CLASSIC_INCLUDES}
)

# Apply compiler flags using our centralized functions
dlp_set_global_compile_flags(dlp_kernel_zen VISIBILITY PRIVATE)
dlp_set_arch_flags(dlp_kernel_zen zen)

# Output the found sources (optional, for debugging)
# message(STATUS "Zen kernel sources: ${ZEN_SOURCES}")

# Export the variables to parent scope
set(ZEN_SOURCES ${ZEN_SOURCES} PARENT_SCOPE)
set(ZEN_INCLUDES ${ZEN_INCLUDES} PARENT_SCOPE)
