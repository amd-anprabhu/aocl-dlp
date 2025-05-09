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

# Find the Sphinx documentation generator
#
# This module defines the following variables:
#   SPHINX_FOUND        - True if Sphinx was found
#   SPHINX_EXECUTABLE   - Path to the Sphinx executable
#   SPHINX_VERSION      - Sphinx version string
#
# The module also defines the following imported target:
#   Sphinx::Sphinx      - Target for using Sphinx

# Find sphinx-build executable
find_program(SPHINX_EXECUTABLE
  NAMES sphinx-build sphinx-build-3
  DOC "Path to sphinx-build executable"
)

# Get version information
if(SPHINX_EXECUTABLE)
  execute_process(
    COMMAND "${SPHINX_EXECUTABLE}" --version
    OUTPUT_VARIABLE SPHINX_VERSION_OUTPUT
    ERROR_VARIABLE SPHINX_VERSION_ERROR
    RESULT_VARIABLE SPHINX_VERSION_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(SPHINX_VERSION_RESULT EQUAL 0)
    if(SPHINX_VERSION_OUTPUT MATCHES "sphinx-build ([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(SPHINX_VERSION "${CMAKE_MATCH_1}")
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx
  REQUIRED_VARS SPHINX_EXECUTABLE
  VERSION_VAR SPHINX_VERSION
)

# Create imported target
if(SPHINX_FOUND AND NOT TARGET Sphinx::Sphinx)
  add_executable(Sphinx::Sphinx IMPORTED)
  set_target_properties(Sphinx::Sphinx PROPERTIES
    IMPORTED_LOCATION "${SPHINX_EXECUTABLE}"
  )
endif()

mark_as_advanced(SPHINX_EXECUTABLE)
