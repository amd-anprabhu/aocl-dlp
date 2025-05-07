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

# Test for specific ZnVer Flags using macros
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# Function to check if a specific compiler flag is supported
# Parameters:
#   flag - The flag to test for support
#   variable - The variable to store the result in (will be set to TRUE or FALSE)
function(dlp_check_compiler_flag flag variable)
    # Strip the -march= prefix if present for the flag variable name
    string(REGEX REPLACE "^-march=" "" flag_name ${flag})
    string(REPLACE "+" "x" flag_name ${flag_name})
    string(REPLACE "-" "_" flag_name ${flag_name})
    string(TOUPPER ${flag_name} flag_name)

    # Check if the C compiler supports the flag
    set(c_var "C_SUPPORTS_${flag_name}")
    check_c_compiler_flag("${flag}" ${c_var})

    # Check if the C++ compiler supports the flag
    set(cxx_var "CXX_SUPPORTS_${flag_name}")
    check_cxx_compiler_flag("${flag}" ${cxx_var})

    # Flag is supported if both C and C++ compilers support it
    if(${c_var} AND ${cxx_var})
        set(${variable} TRUE PARENT_SCOPE)
    else()
        set(${variable} FALSE PARENT_SCOPE)
    endif()
endfunction()
