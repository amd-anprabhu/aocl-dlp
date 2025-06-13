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

# Function to glob source files.
# Usage:
#   dlp_glob_sources(<output_variable> [<directory_to_glob_in>])
#   dlp_glob_sources(OUTPUT <output_variable> [DIRECTORY <directory_to_glob_in> | <directory_to_glob_in>])
function(dlp_glob_sources)
    set(local_output_var "")
    set(local_glob_dir ".") # Default to current directory

    # Create a mutable list of all arguments passed to the function
    set(current_args ${ARGV})

    if(NOT current_args)
        message(FATAL_ERROR "dlp_glob_sources: Output variable name or OUTPUT keyword is required.")
    endif()

    list(GET current_args 0 first_arg_value)

    if("${first_arg_value}" STREQUAL "OUTPUT")
        # Named style starting with OUTPUT: dlp_glob_sources(OUTPUT <out_var> ...)
        list(POP_FRONT current_args temp_output_keyword) # Consume OUTPUT keyword
        if(NOT current_args)
            message(FATAL_ERROR "dlp_glob_sources: OUTPUT keyword requires a variable name argument.")
        endif()
        list(POP_FRONT current_args local_output_var)   # Consume the output variable name
    else()
        # Positional style for output variable: dlp_glob_sources(<out_var> ...)
        list(POP_FRONT current_args local_output_var)   # Consume the output variable name
    endif()

    # At this point, local_output_var is set.
    # current_args contains remaining arguments, which could be:
    # - Empty (glob current directory)
    # - [<directory_path>] (positional directory)
    # - [DIRECTORY <directory_path>] (named directory)

    # Define keywords for cmake_parse_arguments. Using DIRECTORY for directory as requested.
    set(options "")
    set(oneValueArgs "DIRECTORY")
    set(multiValueArgs "")

    cmake_parse_arguments(PARSE_ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${current_args})

    if(DEFINED PARSE_ARGS_DIRECTORY)
        set(local_glob_dir ${PARSE_ARGS_DIRECTORY})
    elseif(PARSE_ARGS_UNPARSED_ARGUMENTS)
        # If DIRECTORY keyword was not used, the first unparsed argument is treated as a positional directory
        list(GET PARSE_ARGS_UNPARSED_ARGUMENTS 0 positional_dir_candidate)
        set(local_glob_dir ${positional_dir_candidate})
        list(REMOVE_AT PARSE_ARGS_UNPARSED_ARGUMENTS 0) # Consume this argument
    endif()

    if(PARSE_ARGS_UNPARSED_ARGUMENTS)
        message(WARNING "dlp_glob_sources: Ignoring leftover unparsed arguments: ${PARSE_ARGS_UNPARSED_ARGUMENTS}")
    endif()

    if(NOT local_output_var)
        message(FATAL_ERROR "dlp_glob_sources: Output variable name could not be determined.")
    endif()

    set(source_files "")

    # Glob all C files
    foreach(ext ${C_EXTENSIONS})
        file(GLOB c_files "${local_glob_dir}/*${ext}")
        list(APPEND source_files ${c_files})
    endforeach()

    # Glob all C++ files
    foreach(ext ${CXX_EXTENSIONS})
        file(GLOB cxx_files "${local_glob_dir}/*${ext}")
        list(APPEND source_files ${cxx_files})
    endforeach()

    # Set the output variable in the parent scope
    set(${local_output_var} ${source_files} PARENT_SCOPE)
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
