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

# Define compiler flags using interface libraries
# This is the modern CMake way to handle compiler flags for Windows/MSVC

# Generic flags for MSVC compiler
set(DLP_GENERIC_FLAGS_MSVC
    /W4             # Warning level 4
    /permissive-    # Standards conformance
    /Zc:inline      # Remove unreferenced functions during optimization
    /Zc:wchar_t     # wchar_t is a native type
)

# Release flags for MSVC compiler
set(DLP_RELEASE_FLAGS_MSVC
    /O2             # Optimize for speed
    /GL             # Whole program optimization
    /Gy             # Function-level linking
    /Oi             # Generate intrinsic functions
    /WX             # Treat warnings as errors (equivalent to -Werror)
)

# Debug flags for MSVC compiler
set(DLP_DEBUG_FLAGS_MSVC
    /Od             # Disable optimization
    /Zi             # Generate complete debugging information
    /RTC1           # Run-time error checks
)

# Architecture-specific flags for MSVC
set(DLP_ARCH_ZEN_FLAGS_MSVC /arch:AVX2)   # AVX2 for Zen architecture
set(DLP_ARCH_ZEN4_FLAGS_MSVC /arch:AVX512) # AVX512 for Zen4 architecture

# Generic flags for non-MSVC compilers (MinGW/Clang) when used on Windows
set(DLP_GENERIC_FLAGS_OTHER -Wall -pedantic)
set(DLP_RELEASE_FLAGS_OTHER -Werror -O3)
set(DLP_DEBUG_FLAGS_OTHER -g -O0)
set(DLP_ARCH_ZEN_FLAGS_OTHER -march=znver2)
set(DLP_ARCH_ZEN4_FLAGS_OTHER -march=znver4)

# Create interface libraries for different flag sets
add_library(dlp_compiler_flags INTERFACE)
add_library(dlp_compiler_flags_release INTERFACE)
add_library(dlp_compiler_flags_debug INTERFACE)

# Set default compiler flags based on compiler
target_compile_options(dlp_compiler_flags INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:${DLP_GENERIC_FLAGS_MSVC}>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${DLP_GENERIC_FLAGS_OTHER}>
)

# Set release-specific compiler flags
target_compile_options(dlp_compiler_flags_release INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:${DLP_RELEASE_FLAGS_MSVC}>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${DLP_RELEASE_FLAGS_OTHER}>
)

# Set debug-specific compiler flags
target_compile_options(dlp_compiler_flags_debug INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:${DLP_DEBUG_FLAGS_MSVC}>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${DLP_DEBUG_FLAGS_OTHER}>
)

# Additional MSVC-specific settings
target_compile_options(dlp_compiler_flags INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/MP>  # Enable multi-processor compilation
)

# Disable specific warnings for MSVC that might be too noisy
target_compile_options(dlp_compiler_flags INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4996>  # Disable deprecation warnings
)

# Function to apply global compiler flags to a target
# Parameters:
#   target - The target to apply flags to
#   visibility - Optional: The visibility level (INTERFACE, PUBLIC, PRIVATE) - defaults to PUBLIC
function(dlp_set_global_compile_flags target)
    # Parse arguments to allow specifying visibility
    set(options "")
    set(oneValueArgs VISIBILITY)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default visibility to PUBLIC if not specified
    if(NOT ARG_VISIBILITY)
        set(ARG_VISIBILITY PUBLIC)
    endif()

    # Apply compiler flags with specified visibility
    target_link_libraries(${target} ${ARG_VISIBILITY}
        dlp_compiler_flags
        $<$<CONFIG:Release>:dlp_compiler_flags_release>
        $<$<CONFIG:RelWithDebInfo>:dlp_compiler_flags_release>
        $<$<CONFIG:Debug>:dlp_compiler_flags_debug>
    )

    # MSVC-specific link options
    if(MSVC)
        # Enable Link Time Optimization in release builds
        set_target_properties(${target} PROPERTIES
            LINK_FLAGS_RELEASE "/LTCG"
            LINK_FLAGS_RELWITHDEBINFO "/LTCG"
        )
    endif()
endfunction()

# Function to set architecture-specific flags for a target
# Parameters:
#   target - The target to apply flags to
#   arch - The architecture (zen, zen4)
#   visibility - Optional: The visibility level (defaults to PRIVATE)
function(dlp_set_arch_flags target arch)
    set(options "")
    set(oneValueArgs VISIBILITY)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default visibility to PRIVATE if not specified
    if(NOT ARG_VISIBILITY)
        set(ARG_VISIBILITY PRIVATE)
    endif()

    # Apply the appropriate architecture flags based on compiler
    if(arch STREQUAL "zen")
        target_compile_options(${target} ${ARG_VISIBILITY}
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<CXX_COMPILER_ID:MSVC>>:${DLP_ARCH_ZEN_FLAGS_MSVC}>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${DLP_ARCH_ZEN_FLAGS_MSVC}>
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:${DLP_ARCH_ZEN_FLAGS_OTHER}>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:${DLP_ARCH_ZEN_FLAGS_OTHER}>
        )
    elseif(arch STREQUAL "zen4")
        target_compile_options(${target} ${ARG_VISIBILITY}
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<CXX_COMPILER_ID:MSVC>>:${DLP_ARCH_ZEN4_FLAGS_MSVC}>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${DLP_ARCH_ZEN4_FLAGS_MSVC}>
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:${DLP_ARCH_ZEN4_FLAGS_OTHER}>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:${DLP_ARCH_ZEN4_FLAGS_OTHER}>
        )
    else()
        message(WARNING "Unknown architecture: ${arch}")
    endif()
endfunction()

# Define Windows-specific functions for build types
function(dlp_set_release_build_flags)
    # Set global release build flags
    if(MSVC)
        set(CMAKE_C_FLAGS_RELEASE "/MD" PARENT_SCOPE)  # Multi-threaded DLL runtime
        set(CMAKE_CXX_FLAGS_RELEASE "/MD" PARENT_SCOPE)
    else()
        set(CMAKE_C_FLAGS_RELEASE "" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS_RELEASE "" PARENT_SCOPE)
    endif()
endfunction()

function(dlp_set_debug_build_flags)
    # Set global debug build flags
    if(MSVC)
        set(CMAKE_C_FLAGS_DEBUG "/MDd" PARENT_SCOPE)  # Multi-threaded Debug DLL runtime
        set(CMAKE_CXX_FLAGS_DEBUG "/MDd" PARENT_SCOPE)
    else()
        set(CMAKE_C_FLAGS_DEBUG "" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS_DEBUG "" PARENT_SCOPE)
    endif()
endfunction()

function(dlp_set_relwithdebinfo_build_flags)
    # Set global relwithdebinfo build flags
    if(MSVC)
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "/MD /Zi" PARENT_SCOPE)  # Multi-threaded DLL runtime with debug info
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi" PARENT_SCOPE)
    else()
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" PARENT_SCOPE)
    endif()
endfunction()
