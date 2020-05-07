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

"""Preprocessing for AST graph representation."""

import concurrent.futures
import os
import shutil
import subprocess
import sys
import tqdm

import representations.utils as app_utils
import utils


def process_source_file(src_file, additional_args=[], is_opencl_source=True):
    """
    Runs graph extractor on a single source file.

    Args:
        src_file: A string representing a source file location.
        additional_args: A list of strings representing additional arguments.
        is_opencl_source: A boolean indicating whether it is OpenCL or not.

    Returns:
        A tuple (stdout, stderr, result) with the extractor's outcome.
    """
    cmd = [app_utils.CLANG_MINER_EXECUTABLE]

    if is_opencl_source:
        cmd += ['-extra-arg-before=-xcl',
               '-extra-arg=-I' + app_utils.LIBCLC_DIR]
        cmd += ['-extra-arg=-include' + app_utils.OPENCL_SHIM_FILE]

    else:
        cmd += ['-extra-arg=-I/Library/Developer/CommandLineTools/usr/include/c++/v1/',
                '-extra-arg=-I/devel/git/llvm_build/build/lib/clang/8.0.0/include/']

    cmd += additional_args
    cmd += [src_file]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    result = process.wait()

    return stdout, stderr, result


def process_source_buffer(src_buffer, additional_args=[], is_opencl_source=True):
    """
    Runs graph extractor on C code contained in a buffer.

    Args:
        src_buffer: A string containing C code.
        additional_args: A list of strings representing additional arguments.
        is_opencl_source: A boolean indicating whether it is OpenCL or not.

    Returns:
        A tuple (stdout, stderr, result) with the extractor's outcome.
    """
    temp_file = '/tmp/temp.c'
    with open(temp_file, 'w') as f:
        f.write(src_buffer)

    return process_source_file(temp_file, additional_args, is_opencl_source)


def process_source_directory(files, preprocessing_artifact_dir, substract_str, is_opencl_source=True):
    """
    Runs graph extractor on a list of files.

    Args:
        files: A list of file path strings.
        preprocessing_artifact_dir: A string of the artifact directory.
        substract_str: A substitution string.
        is_opencl_source: A boolean indicating whether it is OpenCL or not.
    """
    out_dir = os.path.join(preprocessing_artifact_dir, 'out')
    good_code_dir = os.path.join(preprocessing_artifact_dir, 'bad_code')
    bad_code_dir = os.path.join(preprocessing_artifact_dir, 'good_code')
    error_log_dir = os.path.join(preprocessing_artifact_dir, 'error_logs')

    def fnc(filename):
        if substract_str:
            out_filename = filename.replace(substract_str + '/', '') + '.json'
            out_filename = os.path.join(out_dir, out_filename)

            utils.create_folder(os.path.dirname(out_filename))
        else:
            out_filename = filename

        additional_args = []
        if is_opencl_source:
            additional_args += ['-extra-arg-before=-xcl',
                                '-extra-arg=-I' + app_utils.LIBCLC_DIR]
            additional_args += ['-extra-arg=-include' + app_utils.OPENCL_SHIM_FILE]
            if 'npb' in filename:
                additional_args += ['-extra-arg=-DM=1']
            if 'nvidia' in filename or ('rodinia' in filename and 'pathfinder' not in filename):
                additional_args += ['-extra-arg=-DBLOCK_SIZE=64']

        stdout, stderr, result = process_source_file(filename, additional_args=additional_args)

        with open(out_filename, 'wb') as f:
            f.write(stdout)

        # In case of an error
        if result != 0:
            report_filename = os.path.join(
                error_log_dir,
                filename.replace('/', '_') + '.txt')

            # write error report file containing source, stdout, stderr
            utils.write_error_report_file(filename, report_filename, [stdout], [stderr], result, '')

            shutil.copyfile(filename, os.path.join(bad_code_dir, os.path.basename(filename)))
        else:
            shutil.copyfile(filename, os.path.join(good_code_dir, os.path.basename(filename)))

        return result

    # Process files
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm.tqdm(executor.map(fnc, files), total=len(files), desc='C -> Clang graph files', file=sys.stdout))
