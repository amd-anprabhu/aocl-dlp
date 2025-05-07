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

function(dlp_define_extensions)
    set(C_EXTENSIONS ".c" ".C" PARENT_SCOPE)
    set(CXX_EXTENSIONS ".cc" ".cpp" ".CPP" ".cxx" ".CXX" PARENT_SCOPE)
    set(C_HEADER_EXTENSIONS ".h" ".H" PARENT_SCOPE)
    set(CXX_HEADER_EXTENSIONS ".hpp" ".HPP" PARENT_SCOPE)
endfunction()

function(dlp_glob_sources output_var)

    set(source_files "")

    # Glob all C files
    foreach(ext ${C_EXTENSIONS})
        file(GLOB c_files "*${ext}")
        list(APPEND source_files ${c_files})
    endforeach()

    # Glob all C++ files
    foreach(ext ${CXX_EXTENSIONS})
        file(GLOB cxx_files "*${ext}")
        list(APPEND source_files ${cxx_files})
    endforeach()

    # Set the output variable in the parent scope
    set(${output_var} ${source_files} PARENT_SCOPE)
endfunction()

function(dlp_glob_sources_recursive output_var dir)
    set(source_files "")

    # Glob all C files recursively
    foreach(ext ${C_EXTENSIONS})
        file(GLOB_RECURSE c_files "${dir}/*${ext}")
        list(APPEND source_files ${c_files})
    endforeach()

    # Glob all C++ files recursively
    foreach(ext ${CXX_EXTENSIONS})
        file(GLOB_RECURSE cxx_files "${dir}/*${ext}")
        list(APPEND source_files ${cxx_files})
    endforeach()

    # Set the output variable in the parent scope
    set(${output_var} ${source_files} PARENT_SCOPE)
endfunction()
