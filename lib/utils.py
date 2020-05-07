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

"""Utility functions / classes."""

import json
import os
import shutil
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf


# Constants
LABEL_OFFSET = 20
I_OFFSET = 40

# Enums
#######
class AE:
    GRAPH_IDX, STEP_IDX, ACTION, \
    LAST_ADDED_NODE_ID, LAST_ADDED_NODE_TYPE, \
    ACTIONS, \
    GRAPH, NODE_STATES, ADJ_LIST, ACTION_CURRENT_IDX, ACTION_CURRENT, \
    SKIP_NEXT, \
    SUBGRAPH_START, \
    NUM_NODES, \
    PROBABILITY, \
    NUMS_INCOMING_EDGES_BY_TYPE, \
    KERNEL_NAME \
    = range(0, 17)


# Labels
class L:
    LABEL_0, LABEL_1 = range(LABEL_OFFSET, LABEL_OFFSET + 2)

# Type
class T:
    NODES, EDGES, NODE_VALUES = range(30, 33)

# Inputs
class I:
    AUX_IN_0 = range(LABEL_OFFSET, I_OFFSET + 1)


# Functions
###########
def get_dash():
    return '-' * 40


def print_dash():
    print(get_dash())


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


def graph_to_adjacency_lists(graph, tie_fwd_bkwd, edge_type_filter = []) -> (Dict[int, np.ndarray], Dict[int, Dict[int, int]]):
    adj_lists = defaultdict(list)
    num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
    for src, e, dest in graph:
        fwd_edge_type = e
        if fwd_edge_type not in edge_type_filter and len(edge_type_filter) > 0:
            continue

        adj_lists[fwd_edge_type].append((src, dest))
        num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1

        if tie_fwd_bkwd:
            adj_lists[fwd_edge_type].append((dest, src))
            num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

    final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                       for e, lm in adj_lists.items()}

    return final_adj_lists, num_incoming_edges_dicts_per_type


def pretty_print_dict(d: dict) -> None:
    print(json.dumps(d, indent=2))


def json_keys_to_int(x):
    if isinstance(x, dict):
        return { int(k):v for k,v in x.items() }
    return x


def freeze_dict(d):
    if isinstance(d, dict):
        return frozenset((key, freeze_dict(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(freeze_dict(value) for value in d)
    return d


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def print_df(df, max_rows=100):
    with pd.option_context('display.max_rows', max_rows,
                           'display.max_columns', None,
                           'max_colwidth', 999999):
        print(df)


def get_files_by_extension(dirname, extension):
    filepaths = []

    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.endswith(extension):
                filepaths.append(os.path.join(root, file))

    return sorted(filepaths)


def delete_and_create_folder(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def write_error_report_file(src_filename, report_filename, stdouts, stderrs, returncode, cmd):
    report = ''

    report += 'COMMAND:' + '\n'
    report += get_dash() + '\n'
    report += ' '.join(cmd) + '\n'

    report += 'RETURNCODE:' + '\n'
    report += get_dash() + '\n'
    report += str(returncode) + '\n'

    report += 'SOURCE:' + '\n'
    report += get_dash() + '\n'

    with open(src_filename, 'r') as f:
        try:
            report += f.read() + '\n'
        except:
            pass

    for stdout in stdouts:
        report += 'STDOUT:' + '\n'
        report += get_dash() + '\n'
        report += stdout.decode('utf-8') + '\n'

    for stderr in stderrs:
        if stderr:
            report += 'STDERR:' + '\n'
            report += get_dash() + '\n'
            report += stderr.decode('utf-8') + '\n'

    with open(report_filename, 'w+') as f:
        f.write(report)


def prepare_preprocessing_artifact_dir(base_dir):
    delete_and_create_folder(base_dir)
    delete_and_create_folder(os.path.join(base_dir, 'out'))
    delete_and_create_folder(os.path.join(base_dir, 'bad_code'))
    delete_and_create_folder(os.path.join(base_dir, 'good_code'))
    delete_and_create_folder(os.path.join(base_dir, 'error_logs'))


def min_max_avg(l: list) -> dict:
    return {
        'min': min(l),
        'max': max(l),
        'avg': int(sum(l) / float(len(l)))
    }


# Classes
#########
class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, activation, func_name):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.activation = activation
        self.func_name = func_name
        self.params = self.make_network_params()

    def make_network_params(self) -> dict:
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='%s_W_layer%i' % (self.func_name, i))
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='%s_b_layer%i' % (self.func_name, i))
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            'weights': weights,
            'biases': biases,
        }

        return network_params

    def init_weights(self, shape: tuple):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params['weights'], self.params['biases']):
            hid = tf.matmul(acts, W) + b
            if self.activation == 'relu':
                acts = tf.nn.relu(hid)
            elif self.activation == 'sigmoid':
                acts = tf.nn.sigmoid(hid)
            elif self.activation == 'linear':
                acts = hid
            else:
                raise Exception('Unknown activation function: %s' % self.activation)
        last_hidden = hid
        return last_hidden
