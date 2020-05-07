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

"""Preprocessing for LLVM graph representation."""

import concurrent.futures
import os
import shutil
import subprocess
import sys
import tqdm

import representations.utils as app_utils
import utils


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

    return process_source_file(temp_file, additional_args=additional_args, is_opencl_source=is_opencl_source)

def process_source_file(src_file, out_filename='/tmp/out', additional_args=[], is_opencl_source=True):
    """
    Runs graph extractor on a single source file.

    Args:
        src_file: A string representing a source file location.
        out_filename: A string representing the location of temporary output files.
        additional_args: A list of strings representing additional arguments.
        is_opencl_source: A boolean indicating whether it is OpenCL or not.

    Returns:
        A tuple (stdout, stderr, result) with the extractor's outcome.
    """
    # C -> LLVM IR
    cmd_start = [app_utils.CLANG_EXECUTABLE,
                 '-I' + app_utils.LIBCLC_DIR,
                 '-include', app_utils.OPENCL_SHIM_FILE]

    cmd_end = ['-emit-llvm', '-xcl', '-c', src_file, '-o', out_filename + '.ll']
    cmd_compile = cmd_start + additional_args + cmd_end

    process = subprocess.Popen(cmd_compile, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_compile, stderr_compile = process.communicate()
    result_compile = process.returncode

    # LLVM IR -> Graph
    cmd_miner = [app_utils.OPT_EXECUTABLE,
                 '-load', app_utils.MINER_PASS_SHARED_LIBRARY,
                 '-miner', out_filename + '.ll', '-f', '-o', '/dev/null']

    process = subprocess.Popen(cmd_miner, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout_miner, stderr_miner = process.communicate(stdout_compile)
    result_miner = process.returncode

    return stdout_miner, stderr_miner, result_miner


def process_source_directory(files, preprocessing_artifact_dir, substract_str=None, optimize_for_size=False):
    """
    Runs graph extractor on a list of files.

    Args:
        files: A list of file path strings.
        preprocessing_artifact_dir: A string of the artifact directory.
        substract_str: A substitution string.
        optimize_for_size: A boolean indicating whether to optimize for size or not.
    """
    out_dir = os.path.join(preprocessing_artifact_dir, 'out')
    good_code_dir = os.path.join(preprocessing_artifact_dir, 'bad_code')
    bad_code_dir = os.path.join(preprocessing_artifact_dir, 'good_code')
    error_log_dir = os.path.join(preprocessing_artifact_dir, 'error_logs')

    def fnc(filename):
        if substract_str:
            out_filename = filename.replace(substract_str + '/', '')
            out_filename = os.path.join(out_dir, out_filename)

            utils.create_folder(os.path.dirname(out_filename))
        else:
            out_filename = filename

        # C -> LLVM IR
        cmd_start = [app_utils.CLANG_EXECUTABLE,
                     '-I' + app_utils.LIBCLC_DIR,
                     '-include', app_utils.OPENCL_SHIM_FILE]
        cmd_middle = ['-Oz'] if optimize_for_size else ['-O0']
        if 'npb' in filename:
            cmd_middle += ['-DM=1']
        if 'nvidia' in filename or ('rodinia' in filename and 'pathfinder' not in filename):
            cmd_middle += ['-DBLOCK_SIZE=64']

        cmd_end = ['-emit-llvm', '-xcl', '-c', filename, '-o', out_filename + '.ll']
        cmd_compile = cmd_start + cmd_middle + cmd_end

        process = subprocess.Popen(cmd_compile, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_compile, stderr_compile = process.communicate()
        result_compile = process.returncode

        # LLVM IR -> Graph
        cmd_miner = [app_utils.OPT_EXECUTABLE,
                     '-load', app_utils.MINER_PASS_SHARED_LIBRARY,
                     '-miner', out_filename + '.ll', '-f', '-o', '/dev/null']

        process = subprocess.Popen(cmd_miner, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout_miner, stderr_miner = process.communicate(stdout_compile)
        result_miner = process.returncode

        with open(out_filename + '.json', 'bw') as f:
            f.write(stdout_miner)

        # In case of an error
        result = result_compile != 0 or result_miner != 0
        if result is False:
            report_filename = os.path.join(
                error_log_dir,
                filename.replace('/', '_') + '.txt')

            # write error report file containing source, stdout, stderr
            utils.write_error_report_file(filename, report_filename,
                                          [], [stderr_compile, stderr_miner], result, cmd_compile)

            shutil.copyfile(filename, os.path.join(bad_code_dir, os.path.basename(filename)))
        else:
            shutil.copyfile(filename, os.path.join(good_code_dir, os.path.basename(filename)))

        return result

    # Process files
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm.tqdm(executor.map(fnc, files), total=len(files), desc='C -> LLVM graph files', file=sys.stdout))