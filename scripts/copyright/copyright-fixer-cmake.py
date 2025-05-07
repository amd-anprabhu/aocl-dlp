#!/usr/bin/env python3
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

import os
import sys
import argparse

# Set which files to look for
FILTER_EXTENSIONS = ["CMakeLists.txt"]
FILTER_PATTERNS = ["*.cmake"]
COPYRIGHT_DETECTION_HEADER = "Copyright © Advanced Micro Devices"
EXCLUDE_DIRECTORIES = ["build", ".git"]

# Define the complete copyright header for cmake files
COPYRIGHT_HEADER = """#
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

"""

# Detect Copyright Header
def detect_copyright_header(file_path):
    try:
        with open(file_path, "r") as f:
            content = f.read()
            return COPYRIGHT_DETECTION_HEADER in content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

# Get all files to process recursively
def get_files_to_process(directory):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRECTORIES]

        for file in files:
            full_path = os.path.join(root, file)
            # Check for exact match with filter extensions
            if file in FILTER_EXTENSIONS:
                matching_files.append(full_path)
                continue

            # Check for pattern matches
            for pattern in FILTER_PATTERNS:
                if pattern.startswith("*") and file.endswith(pattern[1:]):
                    matching_files.append(full_path)
                    break

    return matching_files

# Fix Copyright Header
def fix_copyright_header(file_path):
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Only add the header if it's not already there
        if COPYRIGHT_DETECTION_HEADER not in content:
            with open(file_path, "w") as f:
                f.write(COPYRIGHT_HEADER + content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Add copyright headers to CMake files.')
    parser.add_argument('directory', nargs='?', default='.',
                        help='Directory to process (default: current directory)')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Automatically proceed without confirmation')
    return parser.parse_args()

# Main
def main():
    args = parse_arguments()

    print(f"Searching for CMake files in {args.directory}, excluding {EXCLUDE_DIRECTORIES}")
    files = get_files_to_process(args.directory)
    print(f"Found {len(files)} files to process")

    if not files:
        print("No CMake files found to process.")
        return

    # Show which files will be processed
    for file in files:
        print(f"  - {file}")

    if not args.yes:
        confirm = input("Do you want to proceed with adding copyright headers to these files? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation canceled.")
            return
    else:
        print("Automatically proceeding with adding copyright headers (-y flag was used)")

    modified_count = 0
    skipped_count = 0

    for file in files:
        if detect_copyright_header(file):
            print(f"Skipping {file} - already has copyright header")
            skipped_count += 1
        else:
            if fix_copyright_header(file):
                print(f"Added copyright header to {file}")
                modified_count += 1
            else:
                print(f"Failed to modify {file}")

    print(f"\nSummary: {modified_count} files modified, {skipped_count} files skipped (already had header)")
    print(f"Total files processed: {len(files)}")

if __name__ == "__main__":
    main()
