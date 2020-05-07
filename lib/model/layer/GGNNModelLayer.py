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

"""GGNN model layer."""

import numpy as np
import tensorflow as tf

from model.layer.PropagationModelLayer import PropagationModelLayer
from utils import glorot_init


class GGNNModelLayerState(object):
    """Holds the state / weights of a GGNN Layer."""

    def __init__(self, config):
        self.config = config

        h_dim = self.config['gnn_h_size']
        num_edge_types = self.config['num_edge_types']

        self.weights = {}

        edge_weights = tf.Variable(glorot_init([num_edge_types * h_dim, h_dim]),
                                                   name='edge_weights')
        self.weights['edge_weights'] = tf.reshape(edge_weights, [num_edge_types, h_dim, h_dim])

        if self.config['use_edge_bias'] == 1:
            self.weights['edge_biases'] = tf.Variable(np.zeros([num_edge_types, h_dim], dtype=np.float32),
                                                            name='gnn_edge_biases')

        cell_type = self.config['graph_rnn_cell'].lower()
        activation_fun = tf.nn.tanh
        if cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
        elif cell_type == 'cudnncompatiblegrucell':
            import tensorflow.contrib.cudnn_rnn as cudnn_rnn
            cell = cudnn_rnn.CudnnCompatibleGRUCell(h_dim)
        elif cell_type == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
        else:
            raise Exception("Unknown RNN cell type '%s'." % cell_type)
        self.weights['rnn_cells'] = cell


class GGNNModelLayer(PropagationModelLayer):
    """Implementation of the model described in
    Li, Yujia, et al. "Gated graph sequence neural networks."

    Sparse implementation from: https://github.com/microsoft/gated-graph-neural-network-samples"""

    def __init__(self, config, state):
        super().__init__()

        self.config = config
        self.state = state

        num_edge_types = self.config['num_edge_types']

        # Placeholders
        h_dim = self.config['gnn_h_size']
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(num_edge_types)]

        if self.config['use_edge_bias'] == 1:
            self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, num_edge_types],
                                                                              name='num_incoming_edges_per_type')

    def compute_embeddings(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Uses the model layer to process embeddings to new embeddings. All embeddings are in one dimension.
        Propagation is made in one pass with many disconnected graphs.

        Args:
            embeddings: Tensor of shape [v, h].

        Returns:
            Tensor of shape [v, h].
        """
        num_nodes = tf.shape(embeddings, out_type=tf.int32)[0]

        # Get all edge targets (aggregate of typed edges)
        edge_targets = []                                                           # list of tensors of message targets of shape [e]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
            edge_targets_for_one_type = adjacency_list_for_edge_type[:, 1]
            edge_targets.append(edge_targets_for_one_type)
        edge_targets = tf.concat(edge_targets, axis=0)                              # [M]

        # Propagate
        for step in range(self.config['num_timesteps']):
            messages = []                                                           # list of tensors of messages of shape [e, h]
            message_source_states = []                                              # list of tensors of edge source states of shape [e, h]

            # Collect incoming messages per edge type
            for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
                edge_sources = adjacency_list_for_edge_type[:, 0]
                edge_source_states = tf.nn.embedding_lookup(params=embeddings,
                                                            ids=edge_sources)       # [e, h]
                all_messages_for_edge_type = tf.matmul(edge_source_states,
                                                       self.state.weights['edge_weights'][edge_type_idx])  # Shape [e, h]
                messages.append(all_messages_for_edge_type)
                message_source_states.append(edge_source_states)

            messages = tf.concat(messages, axis=0)                                  # [M, h]

            messages = tf.unsorted_segment_sum(data=messages,
                                               segment_ids=edge_targets,
                                               num_segments=num_nodes)              # [v, h]

            if self.config['use_edge_bias'] == 1:
                embeddings += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                        self.state.weights['edge_biases'])

            # pass updated vertex features into RNN cell
            embeddings = self.state.weights['rnn_cells'](messages, embeddings)[1]     # [v, h]

        return embeddings
