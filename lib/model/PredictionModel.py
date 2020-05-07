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

"""Prediction model for AST / LLVM graphs."""

import pickle
import sys
import time
import numpy as np
import tensorflow as tf

import utils
from model.cell.PredictionCell import PredictionCell
from model.cell.PredictionCell import PredictionCellState
from model.layer.EmbeddingLayer import EmbeddingLayer
from model.layer.EmbeddingLayer import EmbeddingLayerState
from model.layer.GGNNModelLayer import GGNNModelLayer
from model.layer.GGNNModelLayer import GGNNModelLayerState


class PredictionModelState(object):
    """Holds the state of the prediction model."""
    def __init__(self, config):
        self.graph = tf.Graph()

        self.best_epoch_weights = None

        seed = config['seed']
        tf.set_random_seed(seed)
        np.random.seed(seed)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=tf_config)

        with self.graph.as_default():
            self.embedding_layer_state = EmbeddingLayerState(config)
            self.ggnn_layer_state = GGNNModelLayerState(config)
            self.prediction_cell_state = PredictionCellState(config)

    def __get_weights(self):
        weights = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            weights[variable.name] = self.sess.run(variable)

        return weights

    def __set_weights(self, weights):
        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in weights:
                    restore_ops.append(variable.assign(weights[variable.name]))
                else:
                    # print('Initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in weights:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            self.sess.run(restore_ops)

    def backup_best_weights(self):
        """Backs up current state of the model."""
        self.best_epoch_weights = self.__get_weights()

    def restore_best_weights(self):
        """Restores best state of the model."""
        if self.best_epoch_weights:
            self.__set_weights(self.best_epoch_weights)

    def save_weights_to_disk(self, path):
        """Saves model weights to given file."""
        data = self.__get_weights()

        with open(path, 'wb') as out_file:
            pickle.dump(data, out_file, pickle.HIGHEST_PROTOCOL)

    def restore_weights_from_disk(self, path):
        """Saves model weights to given file."""
        # print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data = pickle.load(in_file)

        self.__set_weights(data)

    def count_number_trainable_params(self):
        """Counts the number of trainable variables.
        Taken from https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
        """
        tot_nb_params = 0
        for trainable_variable in self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
            current_nb_params = self.get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    def get_nb_params_shape(self, shape):
        """Computes the total number of params for a given shape.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        Taken from https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
        """
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params


class PredictionModel(object):
    """Base class of the Trainer and Predictor."""

    def __init__(self, config, state):
        self.config = config
        self.state = state

        self.ggnn_layers = []
        self.cells = []

        with self.state.graph.as_default():
            self.ops = {}
            self.placeholders = {}

        self.with_gradient_monitoring = True if 'gradient_monitoring' in self.config and self.config['gradient_monitoring'] == 1 else False
        self.with_aux_in = True if 'with_aux_in' in self.config and self.config['with_aux_in'] == 1 else False

        # Dump config to stdout
        # utils.pretty_print_dict(self.config)

        with self.state.graph.as_default():
            # Create and initialize model
            self._make_model(True)
            self._make_train_step()
            self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Init tf model
        """
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.state.sess.run(init_op)

    def _graphs_to_batch_feed_dict(self, graphs: list, graph_sizes: list, is_training: bool) -> dict:
        """Creates feed dict that is the format the tf model is trained with."""
        num_edge_types = self.config['num_edge_types']

        batch_data = {
            # Graph model
            'adjacency_lists': [[] for _ in range(self.config['num_edge_types'])],
            'embeddings_to_graph_mappings': [],
            'labels': [],
            'embeddings_in': [],
            'node_values': [],
        }

        if self.with_aux_in:
            batch_data['aux_in'] = []

        for graph_idx, graph in enumerate(graphs):
            num_nodes = graph_sizes[graph_idx]

            if self.config['use_edge_bias'] == 1:
                batch_data['num_incoming_edges_dicts_per_type'] = []

            # Aux in
            if self.with_aux_in:
                if utils.I.AUX_IN_0 in graph:
                    batch_data['aux_in'].append(graph[utils.I.AUX_IN_0])

            # Labels
            if utils.L.LABEL_0 in graph:
                batch_data['labels'].append(graph[utils.L.LABEL_0])

            # Graph model: Adj list
            adj_lists = graph[utils.AE.ADJ_LIST]
            for idx, adj_list in adj_lists.items():
                if idx >= self.config['num_edge_types']:
                    continue
                batch_data['adjacency_lists'][idx].append(adj_list)

            if self.config['use_edge_bias'] == 1:
                # Graph model: Incoming edge numbers
                num_incoming_edges_dicts_per_type = action[utils.AE.NUMS_INCOMING_EDGES_BY_TYPE] if action else utils.graph_to_adjacency_lists([], self.config['tie_fwd_bkwd'], self.config['edge_type_filter'])[0]
                num_incoming_edges_per_type = np.zeros((num_nodes, num_edge_types))
                for (e_type, num_incoming_edges_per_type_dict) in num_incoming_edges_dicts_per_type.items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_data['num_incoming_edges_dicts_per_type'].append(num_incoming_edges_per_type)

            graph_mappings_all = np.full(num_nodes, graph_idx)
            batch_data['embeddings_to_graph_mappings'].append(graph_mappings_all)

            node_types = utils.get_one_hot(np.array(graph[utils.T.NODES]),
                                           self.config['hidden_size_orig']).astype(float)
            batch_data['embeddings_in'].append(node_types)

            if self.config['use_node_values'] == 1:
                node_values = np.array(graph[utils.T.NODE_VALUES]).astype(float).reshape(-1, 1)
                node_values = node_values / 2147483647

                batch_data['node_values'].append(node_values)

        # Build feed dict
        feed_dict = {}

        # Graph model: Adj list
        for idx, adj_list in enumerate(self.ggnn_layers[0].placeholders['adjacency_lists']):
            feed_dict[adj_list] = np.array(batch_data['adjacency_lists'][idx])
            if len(feed_dict[adj_list]) == 0:
                feed_dict[adj_list] = np.zeros((0, 2), dtype=np.int32)
            else:
                feed_dict[adj_list] = feed_dict[adj_list][0]

        if self.config['use_edge_bias'] == 1:
            # Graph model: Incoming edge numbers
            feed_dict[self.ggnn_layers[0].placeholders['num_incoming_edges_per_type']] \
                = np.concatenate(batch_data['num_incoming_edges_dicts_per_type'], axis=0)

        # Is training
        feed_dict[self.cells[0].placeholders['is_training']] = is_training

        # Aux in
        if self.with_aux_in:
            feed_dict[self.cells[0].placeholders['aux_in']] = batch_data['aux_in']

        # Labels
        if 'labels' in self.cells[0].placeholders:
            feed_dict[self.cells[0].placeholders['labels']] = utils.get_one_hot(np.array(batch_data['labels']), self.config['prediction_cell']['output_dim'])

        # Embeddings
        feed_dict[self.placeholders['embeddings_in']] = np.concatenate(batch_data['embeddings_in'], axis=0)
        feed_dict[self.cells[0].placeholders['embeddings_to_graph_mappings']] \
            = np.concatenate(batch_data['embeddings_to_graph_mappings'], axis=0)

        # Node values
        if self.config['use_node_values'] == 1:
            feed_dict[self.placeholders['node_values']] = np.concatenate(batch_data['node_values'], axis=0)

        return feed_dict

    def _make_model(self, enable_training) -> None:
        """Creates tf model."""
        self.placeholders['embeddings_in'] = tf.placeholder(tf.float32, [None, self.config['hidden_size_orig']], name='embeddings_in')
        if self.config['use_node_values'] == 1:
            self.placeholders['node_values'] = tf.placeholder(tf.float32, [None, 1], name='node_values')
            node_values = self.placeholders['node_values']

        # Create model: Unroll network and wire embeddings
        embeddings_reduced = self.placeholders['embeddings_in']

        # Create embedding layer
        embedding_layer = EmbeddingLayer(self.config, self.state.embedding_layer_state)
        embeddings = embedding_layer.compute_embeddings(embeddings_reduced)

        # Create propagation layer
        ggnn_layer = GGNNModelLayer(self.config, self.state.ggnn_layer_state)
        embeddings = ggnn_layer.compute_embeddings(embeddings)

        if self.config['use_node_values'] == 1:
            embeddings = tf.concat([embeddings, node_values], axis=1)

        # Create prediction cell
        prediction_cell = PredictionCell(self.config, enable_training, self.state.prediction_cell_state, self.with_aux_in)
        prediction_cell.initial_embeddings = embeddings_reduced
        prediction_cell.compute_predictions(embeddings)

        self.ggnn_layers.append(ggnn_layer)
        self.cells.append(prediction_cell)

        if enable_training:
            # Cell losses
            losses = []
            for cell in self.cells:
                losses.append(cell.ops['loss'])
            self.ops['losses'] = losses

            # Regularization loss
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                               if 'bias' not in v.name]) * float(self.config['L2_loss_factor'])

            self.ops['loss'] = tf.reduce_sum(losses) + lossL2

    def _make_train_step(self) -> None:
        """Creates tf train step."""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            trainable_vars = self.state.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
            grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)

            # Clipping
            clipped_grads = []
            for grad, var in grads_and_vars:
                if grad is not None:
                    clipped_grads.append((tf.clip_by_norm(grad, self.config['clamp_gradient_norm']), var))
                else:
                    clipped_grads.append((grad, var))

            # Monitoring
            if self.with_gradient_monitoring:
                self.ops['gradients'] = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grads_and_vars])
                self.ops['clipped_gradients'] = tf.summary.merge([tf.summary.histogram("%s-clipped-grad" % g[1].name, g[0]) for g in clipped_grads])

            # Apply
            self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)

        # Initialize newly-introduced variables:
        self.state.sess.run(tf.local_variables_initializer())

    def _run_batch(self, feed_dict):
        """Trains model with one batch and retrieve result."""
        fetch_list = [self.ops['loss'], self.cells[0].ops['output'], self.ops['train_step']]
        if self.with_gradient_monitoring:
            offset = len(fetch_list)
            fetch_list.extend([self.ops['gradients'], self.ops['clipped_gradients']])

        result = self.state.sess.run(fetch_list, feed_dict=feed_dict)

        return result

    def train(self, graphs_train, graphs_test=None, graphs_valid=None) -> None:
        """
        Trains model with training set in multiple epochs.

        Args:
            graphs_train: A list of graphs as train set.
            graphs_test: A list of graphs as test set.
            graphs_valid: A list of graphs as validation set.
        """
        # Enrich graphs with adj list
        for graph in graphs_train:
            graph[utils.AE.ADJ_LIST] = utils.graph_to_adjacency_lists(graph[utils.T.EDGES], self.config['tie_fwd_bkwd'] == 1)[0]

        # Extract graph sizes
        graph_sizes = []
        for graph in graphs_train:
            graph_sizes.append(len(graph[utils.T.NODES]))

        best_epoch_loss = sys.maxsize
        best_epoch_accuracy = 0
        best_epoch_count = 0

        if graphs_valid:
            # Extract valid labels
            y_valid = []
            for graph in graphs_valid:
                y_valid.append(graph[utils.L.LABEL_0])
            y_valid = np.array(y_valid)

        if graphs_test:
            # Extract test labels
            y_test = []
            for graph in graphs_test:
                y_test.append(graph[utils.L.LABEL_0])
            y_test = np.array(y_test)

        # Run epochs
        for epoch in range(0, self.config['num_epochs']):
            # Training
            # ############################################
            epoch_start_time = time.time()

            # Partition into batches
            batch_size = self.config['batch_size']

            lst = list(zip(graphs_train, graph_sizes))
            np.random.shuffle(lst)
            batches = [lst[i * batch_size:(i + 1) * batch_size] for i in
                       range((len(lst) + batch_size - 1) // batch_size)]

            # Run batches
            training_losses = []
            training_good_predicted_all = []
            epoch_instances_per_secs = []

            for batch in batches:
                start_time = time.time()
                batch_graphs, batch_graph_sizes = zip(*batch)

                batch_graphs = list(batch_graphs)
                batch_graph_sizes = list(batch_graph_sizes)

                # Extract train labels
                y_train = []
                for graph in batch_graphs:
                    y_train.append(graph[utils.L.LABEL_0])
                y_train = np.array(y_train)

                # Build feed dict
                # 1. Graph info
                feed_dict = self._graphs_to_batch_feed_dict(batch_graphs, batch_graph_sizes, True)

                # Run batch
                result = self._run_batch(feed_dict)
                end_time = time.time()

                # Log
                instances_per_sec = len(batch_graphs) / (end_time - start_time)
                epoch_instances_per_secs.append(instances_per_sec)

                training_losses.append(result[0])

                # Evaluate predictions on train set
                predictions = result[1]
                training_good_predicted = list(np.argmax(predictions, axis=1) == y_train)
                training_good_predicted_all += training_good_predicted

            training_accuracy = np.sum(training_good_predicted_all) / len(training_good_predicted_all)

            training_loss = np.sum(training_losses)
            epoch_instances_per_sec = np.mean(epoch_instances_per_secs)
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            # Logging
            # summary = tf.Summary()
            # summary.value.add(tag='train_accuracy', simple_value=training_accuracy)
            # summary.value.add(tag='train_loss', simple_value=training_loss)

            # Testing
            # ############################################
            if graphs_valid and graphs_test:
                # Make predictions on valid set
                predictions = self.predict(graphs_valid)

                valid_loss = np.sum(predictions - utils.get_one_hot(y_valid, self.config['prediction_cell']['output_dim']))
                valid_accuracy = np.sum(np.argmax(predictions, axis=1) == y_valid) / len(predictions)

                # Make predictions on test set
                predictions = self.predict(graphs_test)

                test_loss = np.sum(predictions - utils.get_one_hot(y_test, self.config['prediction_cell']['output_dim']))
                test_accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(predictions)

                # Logging
                # summary.value.add(tag='valid_accuracy', simple_value=valid_accuracy)
                # summary.value.add(tag='test_accuracy', simple_value=test_accuracy)
                # print('epoch: %i, instances/sec: %.2f, epoch_time: %.2fs, train_loss: %.8f, train_accuracy: %.4f, valid_accuracy: %.4f, test_accuracy: %.4f' % (epoch, epoch_instances_per_sec, epoch_time, training_loss, training_accuracy, valid_accuracy, test_accuracy))


                if valid_accuracy > best_epoch_accuracy:
                    best_epoch_accuracy = valid_accuracy

                    best_epoch_count += 1
                    if 'save_best_model_interval' in self.config and best_epoch_count >= self.config['save_best_model_interval']:
                        self.state.backup_best_weights()

                        best_epoch_count = 0

            elif graphs_test:
                # Make predictions on test set
                predictions = self.predict(graphs_test)

                test_loss = np.sum(predictions - utils.get_one_hot(y_test, self.config['prediction_cell']['output_dim']))
                test_accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(predictions)

                # Logging
                # summary.value.add(tag='test_accuracy', simple_value=test_accuracy)
                # summary.value.add(tag='test_loss', simple_value=test_loss)
                # print('epoch: %i, instances/sec: %.2f, epoch_time: %.2fs, train_loss: %.8f, train_accuracy: %.4f, test_accuracy: %.4f' % (epoch, epoch_instances_per_sec, epoch_time, training_loss, training_accuracy, test_accuracy))

                if training_loss < best_epoch_loss:
                    best_epoch_loss = training_loss

                    best_epoch_count += 1
                    if 'save_best_model_interval' in self.config and best_epoch_count >= self.config['save_best_model_interval']:
                        self.state.backup_best_weights()

                        best_epoch_count = 0

            else:
                # Logging
                # print('epoch: %i, instances/sec: %.2f, epoch_time: %.2fs, loss: %.8f' % (epoch, epoch_instances_per_sec, epoch_time, training_loss))

                if training_loss < best_epoch_loss:
                    best_epoch_loss = training_loss

                    best_epoch_count += 1
                    if 'save_best_model_interval' in self.config and best_epoch_count >= self.config['save_best_model_interval']:
                        self.state.backup_best_weights()

                        best_epoch_count = 0

            # Logging
            #self.train_writer.add_summary(summary, epoch)

        self.state.restore_best_weights()

    def predict(self, graphs):
        """
        Performs predictions.

        Args:
            graphs: A list of graphs to perform predictions with.

        Returns:
            A list of nums, representing the result.
        """

        # Enrich graphs with adj list
        for graph in graphs:
            graph[utils.AE.ADJ_LIST] = utils.graph_to_adjacency_lists(graph[utils.T.EDGES], self.config['tie_fwd_bkwd'] == 1)[0]

        # Extract graph sizes
        graph_sizes = []
        for graph in graphs:
            graph_sizes.append(len(graph[utils.T.NODES]))

        # Graph info
        feed_dict = self._graphs_to_batch_feed_dict(graphs, graph_sizes, False)

        # Run
        fetch_list = [self.cells[0].ops['output']]
        result = self.state.sess.run(fetch_list, feed_dict=feed_dict)

        return result[0]
