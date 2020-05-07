# Copyright (c) 2019 TU Dresden
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Utility functions / classes for representations."""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def print_process_stdout_continuously(process, prefix):
    """
    Prints stdout without waiting for completion.

    Args:
        process: A subprocess process object.
        prefix: A string to prefix each line with.
    """
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(prefix + ': ' + str(line.rstrip(), 'utf-8'))

    process.wait()
    print(prefix + ' RETURNCODE: ' + str(process.returncode))


def build_with_cmake(project_path, target, additional_cmake_arguments: list = [], is_library=False):
    """
    Builds project with cmake.

    Args:
        project_path: A string representing the path of the project. Must contain a CMakeLists.txt
        target: CMake target to build.
        additional_cmake_arguments: A list of strings, representing addionional cmake arguments.
        is_library: Boolean indicating whether this is a library or not.
    """
    target_executable = 'NOT_BUILT'
    try:
        build_path = os.path.join(project_path, 'build')

        if not os.path.exists(build_path):
            subprocess.Popen(['mkdir', build_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=project_path)

            process = subprocess.Popen(['cmake', '..'] + additional_cmake_arguments,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       cwd=build_path)
            # print_process_stdout_continuously(process, 'CMAKE')

        process = subprocess.Popen(['make', '-j', '4', target], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=build_path)
        # print_process_stdout_continuously(process, 'MAKE')

        if is_library:
            target_executable = os.path.join(build_path, 'lib', 'lib' + target + '.so')
        else:
            target_executable = os.path.join(build_path, target)
    except:
        print(sys.exc_info()[0])

    return target_executable


def get_llvm_build_dir():
    """Get directory where LLVM project is build wither from the PATH or default."""
    # First check for custom provided build
    env_var_name = 'LLVM_BUILD'
    if env_var_name in os.environ:
        return os.environ[env_var_name]

    # Then default to project build
    return os.path.join(SCRIPT_DIR, '../../dependencies/llvm-project/build')


# Executables / Libraries
CLANG_EXECUTABLE = os.path.join(get_llvm_build_dir(), 'bin/clang')
CLANG_FORMAT_EXECUTABLE = os.path.join(get_llvm_build_dir(), 'bin/clang-format')
OPT_EXECUTABLE = os.path.join(get_llvm_build_dir(), 'bin/opt')

MINER_PASS_SHARED_LIBRARY = \
    build_with_cmake(os.path.join(SCRIPT_DIR, '../c/miner_llvm_pass'),
                     'miner_pass',
                     ['-DCMAKE_PREFIX_PATH=' + get_llvm_build_dir() + '/lib/cmake/llvm'], is_library=True)
CLANG_MINER_EXECUTABLE = \
    build_with_cmake(os.path.join(SCRIPT_DIR, '../c/clang_miner'),
                     'clang_miner',
                     ['-DCMAKE_PREFIX_PATH=' + get_llvm_build_dir() + '/lib/cmake/clang'])

LIBCLC_DIR = \
    os.path.join(SCRIPT_DIR, '..', 'c', '3rd_party', 'libclc')
OPENCL_SHIM_FILE = \
    os.path.join(SCRIPT_DIR, '..', 'c', '3rd_party', 'opencl-shim.h')
