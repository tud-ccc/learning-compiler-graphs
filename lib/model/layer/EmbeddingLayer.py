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

"""Embedding layer."""

import tensorflow as tf

import utils
from model.layer.PropagationModelLayer import PropagationModelLayer


class EmbeddingLayerState(object):
    """Holds the state / weights of a Embedding Layer."""

    def __init__(self, config):
        self.config = config

        hidden_size_orig = self.config['hidden_size_orig']
        h_size = self.config['gnn_h_size']
        num_edge_types = self.config['num_edge_types']

        self.weights = {}

        self.weights['mapping'] = utils.MLP(hidden_size_orig, h_size, self.config['embedding_layer']['mapping_dims'], 'relu', 'mapping')


class EmbeddingLayer(PropagationModelLayer):
    """Implementation of the Embedding Layer."""

    def __init__(self, config, state):
        super().__init__()

        self.config = config
        self.state = state

    def compute_embeddings(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Uses the model layer to process embeddings to new embeddings. All embeddings are in one dimension.
        Propagation is made in one pass with many disconnected graphs.

        Args:
            embeddings: Tensor of shape [v, h].

        Returns:
            Tensor of shape [v, h].
        """
        embeddings_new = self.state.weights['mapping'](embeddings)                    # [v, h]

        return embeddings_new