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

"""Thread coarsening factor prediction experiment."""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
import utils
import representations.clang.codegraph_models as clang_codegraph_models
import representations.clang.preprocess as clang_preprocess
import representations.llvm.codegraph_models as llvm_codegraph_models
import representations.llvm.preprocess as llvm_preprocess
from model.PredictionModel import PredictionModel
from model.PredictionModel import PredictionModelState


# Implementation of DeepTune and Magni models, and evaluation copied from:
# https://github.com/ChrisCummins/paper-end2end-dl/blob/master/code/Case%20Study%20A.ipynb
#########################################################
class CLgenError(Exception):
  """Top level error. Never directly thrown."""
  pass


class CLgenObject(object):
    """
    Base object for CLgen classes.
    """
    pass


# Taken from the C99 spec, OpenCL spec 1.2, and bag-of-words analysis of
# GitHub corpus:
OPENCL_ATOMS = set([
    '  ',
    '__assert',
    '__attribute',
    '__builtin_astype',
    '__clc_fabs',
    '__clc_fma',
    '__constant',
    '__global',
    '__inline',
    '__kernel',
    '__local',
    '__private',
    '__read_only',
    '__read_write',
    '__write_only',
    '*/',
    '/*',
    '//',
    'abs',
    'alignas',
    'alignof',
    'atomic_add',
    'auto',
    'barrier',
    'bool',
    'break',
    'case',
    'char',
    'clamp',
    'complex',
    'const',
    'constant',
    'continue',
    'default',
    'define',
    'defined',
    'do',
    'double',
    'elif',
    'else',
    'endif',
    'enum',
    'error',
    'event_t',
    'extern',
    'fabs',
    'false',
    'float',
    'for',
    'get_global_id',
    'get_global_size',
    'get_local_id',
    'get_local_size',
    'get_num_groups',
    'global',
    'goto',
    'half',
    'if',
    'ifdef',
    'ifndef',
    'image1d_array_t',
    'image1d_buffer_t',
    'image1d_t',
    'image2d_array_t',
    'image2d_t',
    'image3d_t',
    'imaginary',
    'include',
    'inline',
    'int',
    'into',
    'kernel',
    'line',
    'local',
    'long',
    'noreturn',
    'pragma',
    'private',
    'quad',
    'read_only',
    'read_write',
    'register',
    'restrict',
    'return',
    'sampler_t',
    'short',
    'shuffle',
    'signed',
    'size_t',
    'sizeof',
    'sqrt',
    'static',
    'struct',
    'switch',
    'true',
    'typedef',
    'u32',
    'uchar',
    'uint',
    'ulong',
    'undef',
    'union',
    'unsigned',
    'void',
    'volatile',
    'while',
    'wide',
    'write_only',
])


class VocabError(CLgenError):
    """A character sequence is not in the atomizer's vocab"""
    pass


class Atomizer(CLgenObject):
    """
    Atomizer.
    """
    def __init__(self, vocab: dict):
        """
        Arguments:
            vocab (dict): A dictionary of string -> integer mappings to use for
                atomizing text from atoms into indices.
        """
        assert(isinstance(vocab, dict))
        self.vocab = vocab
        self._vocab_update()

    @property
    def atoms(self):
        return list(sorted(self.vocab.keys()))

    @property
    def indices(self):
        return list(sorted(self.vocab.values()))

    def _vocab_update(self):
        """ call this when vocab is modified """
        self.vocab_size = len(self.vocab)
        self.decoder = dict((val, key) for key, val in self.vocab.items())

    def atomize(self, text: str) -> np.array:
        """
        Atomize a text into an array of vocabulary indices.
        Arguments:
            text (str): Input text.
        Returns:
            np.array: Indices into vocabulary for all atoms in text.
        """
        raise NotImplementedError("abstract class")

    def tokenize(self, text: str) -> list:
        """
        Atomize a text into an array of atomsself.
        Arguments:
            text (str): Input text.
        Returns:
            list of str: Atom strings.
        """
        indices = self.atomize(text)
        return list(map(lambda x: self.decoder[x], indices))

    def deatomize(self, encoded: np.array) -> str:
        """
        Translate atomized code back into a string.
        Arguments:
            encoded (np.array): Encoded vocabulary indices.
        Returns:
            str: Decoded text.
        """
        try:
            return ''.join(list(map(lambda x: self.decoder[x], encoded)))
        except KeyError:
            raise VocabError

    @staticmethod
    def from_text(text: str):
        """
        Instantiate and specialize an atomizer from a corpus text.
        Arguments:
            text (str): Text corpus
        Returns:
            Atomizer: Specialized atomizer.
        """
        raise NotImplementedError("abstract class")


class CharacterAtomizer(Atomizer):
    """
    An atomizer for character-level syntactic modelling.
    """
    def __init__(self, *args, **kwargs):
        super(CharacterAtomizer, self).__init__(*args, **kwargs)

    def atomize(self, text: str) -> np.array:
        try:
            return np.array(list(map(lambda x: self.vocab[x], text)))
        except KeyError:
            raise VocabError

    def __repr__(self):
        return "CharacterAtomizer[{n} chars]".format(n=self.vocab_size)

    @staticmethod
    def from_text(text: str) -> Atomizer:
        counter = Counter(text)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        atoms, _ = zip(*count_pairs)
        vocab = dict(zip(atoms, range(len(atoms))))
        return CharacterAtomizer(vocab)


class GreedyAtomizer(Atomizer):
    """
    Greedy encoding for multi-characten modelling.
    """
    def __init__(self, *args, **kwargs):
        self.determine_chars = kwargs.pop("determine_chars", False)
        super(GreedyAtomizer, self).__init__(*args, **kwargs)

        multichars = set(k for k in self.atoms if len(k) > 1)
        first_chars = set(a[0] for a in multichars)
        self.lookup = dict((c, [a for a in multichars if a[0] == c])
                           for c in first_chars)

    def atomize(self, text: str) -> np.array:
        def _add_to_vocab(token: str):
            if self.determine_chars and token not in self.vocab:
                maxind = max(self.vocab.values())
                self.vocab[token] = maxind + 1

            return self.vocab[token]

        indices = []
        i = 0
        j = 2
        try:
            while i < len(text):
                if self.lookup.get(text[i]):
                    if j <= len(text) and any(x.startswith(text[i:j])
                                              for x in self.lookup[text[i]]):
                        j += 1
                    else:
                        while j > i + 1:
                            if any(x == text[i:j]
                                   for x in self.lookup[text[i]]):
                                indices.append(self.vocab[text[i:j]])
                                i = j
                                j = j + 2
                                break
                            else:
                                j -= 1
                        else:
                            indices.append(_add_to_vocab(text[i]))
                            i = i + 1
                            j = j + 2
                else:
                    indices.append(_add_to_vocab(text[i]))
                    i = i + 1
                    j = j + 2
        except KeyError:
            raise VocabError

        if self.determine_chars:
            self._vocab_update()

        return np.array(indices)

    def __repr__(self):
        return "GreedyAtomizer[{n} tokens]".format(n=self.vocab_size)

    @staticmethod
    def from_text(text: str) -> Atomizer:
        opencl_vocab = dict(zip(OPENCL_ATOMS, range(len(OPENCL_ATOMS))))
        c = GreedyAtomizer(opencl_vocab, determine_chars=True)

        tokens = sorted(list(set(c.tokenize(text))))
        vocab = dict(zip(tokens, range(len(tokens))))
        return GreedyAtomizer(vocab)


#########################################################
cfs = [1, 2, 4, 8, 16, 32]  # thread coarsening factors

def get_onehot(df, platform):
    hot = np.zeros((len(df), len(cfs)), dtype=np.int32)
    for i, cf in enumerate(df[f"cf_{platform}"]):
        hot[i][cfs.index(cf)] = 1

    return hot


def get_y_naturals(df, platform):
    y = []
    for i, cf in enumerate(df[f"cf_{platform}"]):
        y.append(cfs.index(cf))

    return np.array(y)


def get_magni_features(df, oracles, platform):
    """
    Assemble cascading data.
    """
    X_cc, y_cc, = [], []
    for kernel in sorted(set(df["kernel"])):
        _df = df[df["kernel"] == kernel]

        oracle_cf = int(oracles[oracles["kernel"] == kernel][f"cf_{platform}"].values[0])

        feature_vectors = np.asarray([
            _df['PCA1'].values,
            _df['PCA2'].values,
            _df['PCA3'].values,
            _df['PCA4'].values,
            _df['PCA5'].values,
            _df['PCA6'].values,
            _df['PCA7'].values,
        ]).T

        X_cc.append(feature_vectors)
        y = []
        cfs__ = []
        for i, cf in enumerate(cfs[:len(feature_vectors)]):
            y_ = 1 if cf < oracle_cf else 0
            y.append(y_)
        y_cc.append(y)

        assert len(feature_vectors) == len(y)

    assert len(X_cc) == len(y_cc) == 17

    return np.asarray(X_cc), np.asarray(y_cc)


def get_clang_graphs(df: pd.DataFrame) -> np.array:
    return np.array(
        df["clang_graph"].values,
    ).T


def consolidate_by_kernel_name(kernel_names, elements):
    kernels_consolidated = []
    elements_consolidated = []

    for kernel, src in zip(kernel_names, elements):
        if kernel not in kernels_consolidated:
            kernels_consolidated.append(kernel)
            elements_consolidated.append(src)

    return elements_consolidated


def encode_srcs(atomizer, kernels, srcs):
    """ encode and pad source code for learning """
    from keras.preprocessing.sequence import pad_sequences

    # Consolidate srcs
    srcs_consolidated = consolidate_by_kernel_name(kernels, srcs)

    seqs = [atomizer.atomize(src) for src in srcs_consolidated]
    pad_val = atomizer.vocab_size
    encoded = np.array(pad_sequences(seqs, maxlen=1024, value=pad_val))

    np.set_printoptions(threshold=sys.maxsize)

    return np.vstack([np.expand_dims(x, axis=0) for x in encoded])


def platform2str(platform):
    if platform == "Fermi":
        return "NVIDIA GTX 480"
    elif platform == "Kepler":
        return "NVIDIA Tesla K20c"
    elif platform == "Cypress":
        return "AMD Radeon HD 5900"
    elif platform == "Tahiti":
        return "AMD Tahiti 7970"
    else:
        raise LookupError


def evaluate(model, df_runtimes, df_oracles, seed, report_write_dir, run_id):
    from sklearn.model_selection import KFold

    data = []

    X_seq = None  # defer sequence encoding (it's expensive)

    for i, platform in enumerate(["Cypress", "Tahiti", "Fermi", "Kepler"]):
        platform_name = platform2str(platform)

        # load data
        oracle_runtimes = np.array([float(x) for x in df_oracles["runtime_" + platform]])
        y = np.array([int(x) for x in df_oracles["cf_" + platform]], dtype=np.int32)
        y_1hot = get_onehot(df_oracles, platform)
        y_naturals = get_y_naturals(df_oracles, platform)
        X_cc, y_cc = get_magni_features(df_runtimes, df_oracles, platform)

        clang_graphs = np.array(consolidate_by_kernel_name(df_runtimes["kernel"].values, df_runtimes["clang_graph"].values))
        llvm_graphs = np.array(consolidate_by_kernel_name(df_runtimes["kernel"].values, df_runtimes["llvm_graph"].values))

        # LOOCV
        kf = KFold(n_splits=len(y), shuffle=False)

        for j, (train_index, test_index) in tqdm.tqdm(enumerate(kf.split(y)), desc=platform, total=len(oracle_runtimes), file=sys.stdout):
            model.init(seed=seed)

            # Prepare data
            if (model.__class__.__name__ == 'DeepTune' or model.__class__.__name__ == 'Magni') and X_seq is None:  # encode source codes if needed
                X_seq = encode_srcs(model.atomizer, df_runtimes["kernel"].values, df_runtimes["src"].values)

            clang_graphs_train = [json.loads(g, object_hook=utils.json_keys_to_int) for g in
                                            clang_graphs[train_index]]
            clang_graphs_test = [json.loads(g, object_hook=utils.json_keys_to_int) for g in
                                           clang_graphs[test_index]]
            llvm_graphs_train = [json.loads(g, object_hook=utils.json_keys_to_int) for g in
                                            llvm_graphs[train_index]]
            llvm_graphs_test = [json.loads(g, object_hook=utils.json_keys_to_int) for g in
                                            llvm_graphs[test_index]]

            kernel = sorted(set(df_runtimes["kernel"]))[test_index[0]]

            # Train/Test
            model_base_path = os.path.join(report_write_dir, 'models', run_id)
            model_path = os.path.join(model_base_path,  f"{model.__basename__}-{platform}-{j}.model")
            if not os.path.exists(model_base_path):
                os.makedirs(model_base_path)

            if not os.path.exists(model_path):
                print('Training model')

                # train a model
                train_time_start = time.time()
                model.train(cascading_features=np.concatenate(X_cc[train_index]),
                            cascading_y=np.concatenate(y_cc[train_index]),
                            clang_graphs_train=clang_graphs_train,
                            clang_graphs_test=clang_graphs_test,
                            llvm_graphs_train=llvm_graphs_train,
                            llvm_graphs_test=llvm_graphs_test,
                            sequences=X_seq[train_index] if X_seq is not None else None,
                            verbose=True,
                            y_1hot=y_1hot[train_index],
                            y_naturals_train=y_naturals[train_index],
                            y_naturals_test=y_naturals[test_index])
                train_time_end = time.time()
                train_time = train_time_end - train_time_start

                model.save(model_path)

            else:
                # restore a model
                model.restore(model_path)

                train_time = None

            # make prediction
            inference_time_start = time.time()
            p = model.predict(cascading_features=X_cc[test_index[0]],
                              clang_graphs_test=clang_graphs_test,
                              llvm_graphs_test=llvm_graphs_test,
                              sequences=X_seq[test_index] if X_seq is not None else None,
                              y_naturals_test=y_naturals[test_index]
                              )[0]
            inference_time_end = time.time()
            inference_time = inference_time_end - inference_time_start

            p = min(p, 2 ** (len(X_cc[test_index[0]]) - 1))

            # oracle prediction
            o = y[test_index[0]]
            correct = p == o

            # get runtime without thread coarsening
            row = df_runtimes[(df_runtimes["kernel"] == kernel) & (df_runtimes["cf"] == 1)]
            assert (len(row) == 1)  # sanity check
            nocf_runtime = float(row["runtime_" + platform])

            # get runtime of prediction
            row = df_runtimes[(df_runtimes["kernel"] == kernel) & (df_runtimes["cf"] == p)]
            assert (len(row) == 1)  # sanity check
            p_runtime = float(row["runtime_" + platform])

            # get runtime of oracle coarsening factor
            o_runtime = oracle_runtimes[test_index[0]]

            # speedup and % oracle
            s_oracle = nocf_runtime / o_runtime
            p_speedup = nocf_runtime / p_runtime
            p_oracle = o_runtime / p_runtime

            # record result
            data.append({
                "Model": model.__name__,
                "Platform": platform_name,
                "Kernel": kernel,
                "Oracle-CF": o,
                "Predicted-CF": p,
                "Speedup": p_speedup,
                "Oracle": p_oracle,
                "num_trainable_parameters": model.get_num_trainable_parameters(),
                "train_time": train_time,
                "inference_time": inference_time
            })

            model.clear()

    return pd.DataFrame(
        data, columns=[
            "Model",
            "Platform",
            "Kernel",
            "Oracle-CF",
            "Predicted-CF",
            "Speedup",
            "Oracle",
            "num_trainable_parameters",
            "train_time",
            "inference_time"])


class ThreadCoarseningModel(object):
    """
    A model for predicting OpenCL thread coarsening factors.

    Attributes
    ----------
    __name__ : str
        Model name
    __basename__ : str
        Shortened name, used for files
    """
    __name__ = None
    __basename__ = None

    def init(self, seed: int) -> None:
        """
        Initialize the model.

        Do whatever is required to setup a new thread coarsening model here.
        This method is called prior to training and predicting.
        This method may be omitted if no initial setup is required.

        Parameters
        ----------
        seed : int
            The seed value used to reproducible results. May be 'None',
            indicating that no seed is to be used.
        """
        pass

    def save(self, outpath: str) -> None:
        raise NotImplementedError

    def restore(self, inpath: str) -> None:
        raise NotImplementedError

    def train(self, **data) -> None:
        """
        Train a model.

        Parameters
        ----------
        cascading_features : np.array
            An array of feature vectors of shape (n,7,7). Used for the cascading
            model, there are 7 vectors of 7 features for each benchmark, one for
            each coarsening factor.

        cascading_y : np.array
            An array of classification labels of shape(n,7). Used for the cascading
            model.

        sequences : np.array
            An array of encoded source code sequences of shape (n,seq_length).

        y_1hot : np.array
            An array of optimal coarsening factors of shape (n,6), in 1-hot encoding.

        verbose: bool, optional
            Whether to print verbose status messages during training.
        """
        raise NotImplementedError

    def predict(self, **data) -> np.array:
        """
        Make predictions for programs.

        Parameters
        ----------
        cascading_features : np.array
            An array of feature vectors of shape (n,7,7). Used for the cascading
            model, there are 7 vectors of 7 features for each benchmark, one for
            each coarsening factor.

        sequences : np.array
            An array of encoded source code sequences of shape (n,seq_length).

        Returns
        -------
        np.array
            Predicted 'y' values (optimal thread coarsening factors) with shape (n,1).
        """
        raise NotImplementedError

    def get_num_trainable_parameters(self):
        return None

    def clear(self):
        pass


class Magni(ThreadCoarseningModel):
    __name__ = "Magni et al."
    __basename__ = "magni"

    def init(self, seed: int=None):
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import GridSearchCV, KFold

        # during grid search, not all parameters will converge. Ignore these warnings
        from warnings import filterwarnings
        from sklearn.exceptions import ConvergenceWarning
        filterwarnings("ignore", category=ConvergenceWarning)

        # the neural network
        nn = MLPClassifier(random_state=seed, shuffle=True)

        # cross-validation over the training set. We train on 16 programs,
        # so with k=16 and no shuffling of the data, we're performing
        # nested leave-one-out cross-validation
        inner_cv = KFold(n_splits=16, shuffle=False)

        # hyper-parameter combinations to try
        params = {
            "max_iter": [200, 500, 1000, 2000],
            "hidden_layer_sizes": [
                (32,),
                (32, 32),
                (32, 32, 32),
                (64,),
                (64, 64),
                (64, 64, 64),
                (128,),
                (128, 128),
                (128, 128, 128),
                (256,),
                (256, 256),
                (256, 256, 256),
            ]
        }

        self.model = GridSearchCV(nn, cv=inner_cv, param_grid=params, n_jobs=-1, verbose=2)

    def save(self, outpath):
        with open(outpath, 'wb') as outfile:
            pickle.dump(self.model, outfile)

    def restore(self, inpath):
        with open(inpath, 'rb') as infile:
            self.model = pickle.load(infile)

    def train(self, **data) -> None:
        self.model.fit(data['cascading_features'], data['cascading_y'])

    def predict(self, **data) -> np.array:
        # we only support leave-one-out cross-validation (implementation detail):
        assert(len(data['sequences']) == 1)

        # The binary cascading model:
        #
        # iteratively apply thread coarsening, using a new feature vector
        # every time coarsening is applied
        for i in range(len(data['cascading_features'])):
            # predict whether to coarsen, using the program features of
            # the current coarsening level:
            should_coarsen = self.model.predict([data['cascading_features'][i]])[0]
            if not should_coarsen:
                break
        p = cfs[i]
        return [cfs[i]]


class DeepTune(ThreadCoarseningModel):
    __name__ = "DeepTune"
    __basename__ = "deeptune"

    def init(self, seed: int = None):
        from keras.layers import Input, Dropout, Embedding, merge, LSTM, Dense
        from keras.layers.normalization import BatchNormalization
        from keras.models import Model, Sequential, load_model

        np.random.seed(seed)

        # Vocabulary has a padding character
        vocab_size = self.atomizer.vocab_size + 1

        # Language model. Takes as inputs source code sequences.
        seq_inputs = Input(shape=(1024,), dtype="int32")
        x = Embedding(input_dim=vocab_size, input_length=1024,
                      output_dim=64, name="embedding")(seq_inputs)
        x = LSTM(64, return_sequences=True, implementation=1, name="lstm_1")(x)
        x = LSTM(64, implementation=1, name="lstm_2")(x)

        # Heuristic model. Takes as inputs the language model,
        #   outputs 1-of-6 thread coarsening factor
        x = BatchNormalization()(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(6, activation="sigmoid")(x)

        self.model = Model(inputs=seq_inputs, outputs=outputs)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    def save(self, outpath):
        self.model.save(outpath)

    def restore(self, inpath):
        from keras.models import load_model
        self.model = load_model(inpath)

    def train(self, **data) -> None:
        self.model.fit(data['sequences'], data['y_1hot'], epochs=50, batch_size=64, verbose=data['verbose'], shuffle=True)

    def predict(self, **data) -> np.array:
        # directly predict optimal thread coarsening factor from source sequences:
        p = np.array(self.model.predict(data['sequences'], batch_size=64, verbose=0))
        indices = [np.argmax(x) for x in p]
        return [cfs[x] for x in indices]

    def get_num_trainable_parameters(self):
        return self.model.count_params()

    def clear(self):
        import gc
        import keras.backend as K

        # see https://github.com/keras-team/keras/issues/2102
        K.clear_session()
        del self.model
        gc.collect()


class GNN(ThreadCoarseningModel):
    __name__ = "GNN"
    __basename__ = "gnn"

    def __init__(self, config):
        self.config = config

    def init(self, seed):
        self.state = PredictionModelState(self.config)
        self.model = PredictionModel(self.config, self.state)

        # print('Number of trainable parameters:', self.state.count_number_trainable_params())

        return self

    def save(self, outpath):
        self.state.save_weights_to_disk(outpath)

    def restore(self, inpath):
        self.state.restore_weights_from_disk(inpath)

    def train(self, **data):
        prefix = 'clang' if self.__basename__ == 'gnn-ast' else 'llvm'

        graphs_train = []
        for graph, y in zip(data[prefix + "_graphs_train"], data["y_naturals_train"]):
            graph[utils.L.LABEL_0] = y
            graphs_train.append(graph)

        graphs_test = []
        for graph, y in zip(data[prefix + "_graphs_test"], data["y_naturals_test"]):
            graph[utils.L.LABEL_0] = y
            graphs_test.append(graph)

        self.model.train(graphs_train, graphs_test)

    def predict(self, **data):
        prefix = 'clang' if self.__basename__ == 'gnn-ast' else 'llvm'

        graphs_test = []
        for graph, y in zip(data[prefix + "_graphs_test"], data["y_naturals_test"]):
            graph[utils.L.LABEL_0] = y
            graphs_test.append(graph)

        p = self.model.predict(graphs_test)
        p = np.array(p)
        indices = [np.argmax(x) for x in p]

        return [cfs[x] for x in indices]

    def get_num_trainable_parameters(self):
        return self.state.count_number_trainable_params()


class GnnAST(GNN):
    __name__ = "GNN-AST"
    __basename__ = "gnn-ast"

    def __init__(self, config):
        GNN.__init__(self, config)


class GnnLLVM(GNN):
    __name__ = "GNN-LLVM"
    __basename__ = "gnn-llvm"

    def __init__(self, config):
        GNN.__init__(self, config)


def parse_report_to_summary(report: pd.DataFrame):
    report_str = ''

    report_str += 'Grouped by Platform\n'
    report_str += str(report.groupby('Platform')['Platform', 'Speedup', 'Oracle'].mean())
    report_str += '\n\n'

    return report_str


def print_and_save_report(report_write_dir, run_id, config, model, report):
    # Write to files
    # Config
    filename = model.__basename__ + '_' + str(run_id) + '_config.txt'
    with open(os.path.join(report_write_dir, filename), 'w') as f:
        f.write(json.dumps(config))

    # Raw
    filename = model.__basename__ + '_' + str(run_id) + '_raw.txt'
    with open(os.path.join(report_write_dir, filename), 'w') as f:
        f.write(report.to_csv())


def preprocess(**kwargs):
    args = argparse.Namespace(**kwargs)

    df_runtimes = pd.read_csv(args.cgo17_runtimes_csv)

    sample_to_src_mapping = {
        'blackscholes': {
            'kernel_name': 'BlackScholes',
            'src_location': 'nvidia-4.2/OpenCL/src/oclBlackScholes/BlackScholes.cl'
        },
        'mt': {
            'kernel_name': 'matrixTranspose',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/MatrixTranspose/MatrixTranspose_Kernels.cl'
        },
        'mtLocal': {
            'kernel_name': 'matrixTranspose',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/MatrixTranspose/MatrixTranspose_Kernels.cl'
        },
        'sgemm': {
            'kernel_name': 'mysgemmNT',
            'src_location': 'parboil-0.2/benchmarks/sgemm/src/opencl_base/kernel.cl'
        },
        'spmv': {
            'kernel_name': 'A',
            'src_location': 'parboil-0.2/benchmarks/spmv/src/opencl_base/kernel.cl'
        },
        'stencil': {
            'kernel_name': 'naive_kernel',
            'src_location': 'parboil-0.2/benchmarks/stencil/src/opencl_base/kernel.cl'
        },
        'mriQ': {
            'kernel_name': 'ComputeQ_GPU',
            'src_location': 'parboil-0.2/benchmarks/mri-q/src/opencl_base/kernels.cl'
        },
        'mvCoal': {
            'kernel_name': 'MatVecMulCoalesced0',
            'src_location': 'nvidia-4.2/OpenCL/src/oclMatVecMul/oclMatVecMul.cl'
        },
        'floydWarshall': {
            'kernel_name': 'floydWarshallPass',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/FloydWarshall/FloydWarshall_Kernels.cl'
        },
        'fastWalsh': {
            'kernel_name': 'fastWalshTransform',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/FastWalshTransform/FastWalshTransform_Kernels.cl'
        },
        'dwtHaar1D': {
            'kernel_name': 'dwtHaar1D',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/DwtHaar1D/DwtHaar1D_Kernels.cl'
        },
        'convolution': {
            'kernel_name': 'simpleNonSeparableConvolution',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/SimpleConvolution/SimpleConvolution_Kernels.cl'
        },
        'binarySearch': {
            'kernel_name': 'bitonicSort',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/BitonicSort/BitonicSort_Kernels.cl'
        },
        'mvUncoal': {
            'kernel_name': 'MatVecMulUncoalesced0',
            'src_location': 'nvidia-4.2/OpenCL/src/oclMatVecMul/oclMatVecMul.cl'
        },
        'sobel': {
            'kernel_name': 'sobel_filter',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/SobelFilter/SobelFilter_Kernels.cl'
        },
        'reduce': {
            'kernel_name': 'reduce',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/Reduction/Reduction_Kernels.cl'
        },
        'nbody': {
            'kernel_name': 'nbody_sim',
            'src_location': 'amd-app-sdk-3.0/samples/opencl/cl/1.x/NBody/NBody_Kernels.cl'
        }
    }

    # Clang
    # ############################################
    # 1) Extract graphs
    clang_graphs = {}
    for kernel_name_in_cgo17, data in tqdm.tqdm(sample_to_src_mapping.items(), desc='C -> Clang graphs', file=sys.stdout):
        src_file = os.path.join(args.code_dir, data['src_location'])

        miner_out, _, _ = clang_preprocess.process_source_file(src_file)
        graphs = clang_codegraph_models.codegraphs_create_from_miner_output(json.loads(miner_out))
        # Get correct function
        for graph in graphs:
            f = graph.functions[0]
            if f.name == data['kernel_name']:
                # Transform
                graph_transformed = clang_codegraph_models.transform_graph(graph)

                clang_graphs[kernel_name_in_cgo17] = graph_transformed
        assert graph

    # assert len(clang_graphs) == 17

    # 2) Get node type ids
    node_types = clang_codegraph_models.get_node_types(clang_graphs, with_functionnames=False, with_callnames=False)

    clang_info = {
        'num_node_types': len(node_types)
    }

    # 3) Add to dataframe
    df_runtimes['clang_graph'] = None
    for row_idx, row in df_runtimes.iterrows():
        kernel_name = row['kernel']
        if kernel_name in clang_graphs:
            graph = clang_graphs[kernel_name]
            graph_export = clang_codegraph_models.graph_to_export_format(graph)

            df_runtimes.loc[row_idx, 'clang_graph'] = json.dumps(graph_export)

    # LLVM
    # ############################################
    # 1) Extract graphs
    llvm_graphs = {}
    for kernel_name_in_cgo17, data in tqdm.tqdm(sample_to_src_mapping.items(), desc='C -> LLVM graphs', file=sys.stdout):
        src_file = os.path.join(args.code_dir, data['src_location'])
        # LLVM
        miner_out, _, _ = llvm_preprocess.process_source_file(src_file)
        graphs = llvm_codegraph_models.codegraphs_create_from_miner_output(json.loads(miner_out))
        # Get correct function
        for graph in graphs:
            f = graph.functions[0]
            if f.name == data['kernel_name']:
                matched_graph = graph
        assert matched_graph

        llvm_graphs[kernel_name_in_cgo17] = matched_graph

    # assert len(llvm_graphs) == 17

    # 2) Get node type ids
    node_types = llvm_codegraph_models.get_node_types(llvm_graphs)

    llvm_info = {
        'num_node_types': len(node_types)
    }

    # 3) Add to dataframe
    df_runtimes['llvm_graph'] = None
    for row_idx, row in df_runtimes.iterrows():
        kernel_name = row['kernel']
        if kernel_name in llvm_graphs:
            graph = llvm_graphs[kernel_name]
            graph_export = llvm_codegraph_models.graph_to_export_format(graph, node_types)

            df_runtimes.loc[row_idx, 'llvm_graph'] = json.dumps(graph_export)

    # Write to out csv file
    if args.cgo17_runtimes_csv_out:
        df_runtimes.to_csv(args.cgo17_runtimes_csv_out)

    utils.pretty_print_dict({
        'Clang': clang_info,
        'LLVM': llvm_info
    })


def experiment(**kwargs):
    def is_set(obj, attr):
        return hasattr(obj, attr) and getattr(obj, attr)

    args = argparse.Namespace(**kwargs)

    # Load datasets
    df_runtimes = pd.read_csv(args.runtimes_csv)
    df_oracles = pd.read_csv(args.oracles_csv)
    df_devmap_amd = pd.read_csv(args.devmap_amd_csv)

    seed = int(args.seed)
    run_id = str(args.seed)
    config = {}

    if is_set(args, 'magni'):
        model = Magni()

        srcs = '\n'.join(df_devmap_amd['src'].values)
        model.atomizer = GreedyAtomizer.from_text(srcs)

    if is_set(args, 'deeptune'):
        model = DeepTune()

        srcs = '\n'.join(df_devmap_amd['src'].values)
        model.atomizer = GreedyAtomizer.from_text(srcs)

    if is_set(args, 'gnn_ast') or is_set(args, 'gnn_ast_astonly'):
        config = {
            "run_id": 'gnn-ast' + '_' + str(run_id),

            "graph_rnn_cell": "GRU",

            "num_timesteps": 4,
            "hidden_size_orig": 46,
            "gnn_h_size": 4,
            "gnn_m_size": 2,

            "num_edge_types": 2,

            "prediction_cell": {
                "mlp_f_m_dims": [],
                "mlp_f_m_activation": "relu",

                "mlp_g_m_dims": [],
                "mlp_g_m_activation": "sigmoid",

                "mlp_reduce_dims": [],
                "mlp_reduce_activation": "relu",

                "mlp_reduce_after_aux_in_1_dims": [],
                "mlp_reduce_after_aux_in_1_activation": "relu",
                "mlp_reduce_after_aux_in_1_out_dim": 4,

                "mlp_reduce_after_aux_in_2_dims": [],
                "mlp_reduce_after_aux_in_2_activation": "sigmoid",
                "mlp_reduce_after_aux_in_2_out_dim": 6,

                "output_dim": 6,
            },

            "embedding_layer": {
                "mapping_dims": []
            },

            "learning_rate": 0.0005,
            "clamp_gradient_norm": 1.0,
            "L2_loss_factor": 0.01,

            "batch_size": 16,
            "num_epochs": 500,
            "out_dir": "/tmp",

            "tie_fwd_bkwd": 1,
            "use_edge_bias": 0,
            "use_edge_msg_avg_aggregation": 0,

            "use_node_values": 0,
            "save_best_model_interval": 1,
            "with_aux_in": 0,

            "seed": seed
        }

        if is_set(args, 'gnn_ast_astonly'):
            config['edge_type_filter'] = [0]
            config['num_edge_types'] = 1

        model = GnnAST(config)

    if is_set(args, 'gnn_llvm') or is_set(args, 'gnn_llvm_cfgonly') or is_set(args, 'gnn_llvm_cfgdataflowonly') or is_set(args, 'gnn_llvm_cfgdataflowcallonly'):
        config = {
            "run_id": 'gnn-llvm' + '_' + str(run_id),

            "graph_rnn_cell": "GRU",

            "num_timesteps": 4,
            "hidden_size_orig": 54,
            "gnn_h_size": 4,
            "gnn_m_size": 2,

            "num_edge_types": 4,

            "prediction_cell": {
                "mlp_f_m_dims": [],
                "mlp_f_m_activation": "relu",

                "mlp_g_m_dims": [],
                "mlp_g_m_activation": "sigmoid",

                "mlp_reduce_dims": [],
                "mlp_reduce_activation": "relu",

                "mlp_reduce_after_aux_in_1_dims": [],
                "mlp_reduce_after_aux_in_1_activation": "relu",
                "mlp_reduce_after_aux_in_1_out_dim": 4,

                "mlp_reduce_after_aux_in_2_dims": [],
                "mlp_reduce_after_aux_in_2_activation": "sigmoid",
                "mlp_reduce_after_aux_in_2_out_dim": 6,

                "output_dim": 6,
            },

            "embedding_layer": {
                "mapping_dims": []
            },

            "learning_rate": 0.0005,
            "clamp_gradient_norm": 1.0,
            "L2_loss_factor": 0,

            "batch_size": 16,
            "num_epochs": 500,
            "out_dir": "/tmp",

            "tie_fwd_bkwd": 1,
            "use_edge_bias": 0,
            "use_edge_msg_avg_aggregation": 0,

            "use_node_values": 0,
            "save_best_model_interval": 1,
            "with_aux_in": 0,

            "seed": seed
        }

        if is_set(args, 'gnn_llvm_cfgonly'):
            config['edge_type_filter'] = [0]
            config['num_edge_types'] = 1

        if is_set(args, 'gnn_llvm_cfgdataflowonly'):
            config['edge_type_filter'] = [0, 1]
            config['num_edge_types'] = 2

        if is_set(args, 'gnn_llvm_cfgdataflowcallonly'):
            config['edge_type_filter'] = [0, 1, 2]
            config['num_edge_types'] = 3

        model = GnnLLVM(config)

    print("\nTraining/Predicting %s with seed %i ..." % (model.__name__, seed))

    report = evaluate(model, df_runtimes, df_oracles, seed, args.report_write_dir, run_id)
    print_and_save_report(args.report_write_dir, run_id, config, model, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='Subcommand to run')
    subparsers = parser.add_subparsers()

    # Parse command
    command_arg = parser.parse_args(sys.argv[1:2])
    if not hasattr(command_arg, 'command'):
        print('Unrecognized command')
        parser.print_help()
        exit(1)

    # Experiment command
    if command_arg.command == 'experiment':
        # Parse args
        parser_exp = subparsers.add_parser('train')

        parser_exp.add_argument('--magni', action='store_true')
        parser_exp.add_argument('--deeptune', action='store_true')
        parser_exp.add_argument('--gnn_ast', action='store_true')
        parser_exp.add_argument('--gnn_ast_astonly', action='store_true')
        parser_exp.add_argument('--gnn_llvm', action='store_true')
        parser_exp.add_argument('--gnn_llvm_cfgonly', action='store_true')
        parser_exp.add_argument('--gnn_llvm_cfgdataflowonly', action='store_true')
        parser_exp.add_argument('--gnn_llvm_cfgdataflowcallonly', action='store_true')

        parser_exp.add_argument('--runtimes_csv')
        parser_exp.add_argument('--oracles_csv')
        parser_exp.add_argument('--devmap_amd_csv')

        parser_exp.add_argument('--seed')
        parser_exp.add_argument('--report_write_dir')

        args = parser_exp.parse_args(sys.argv[2:])

        experiment(**vars(args))

    # Preprocess command
    if command_arg.command == 'preprocess':
        # Parse args
        parser_prep = subparsers.add_parser('preprocess')

        parser_prep.add_argument("--code_dir", type=str,
                                 help="directory of c code files")
        parser_prep.add_argument("--preprocessing_artifact_dir", type=str,
                                 help="out directory containing preprocessing information")
        parser_prep.add_argument('--cgo17_runtimes_csv', type=str)
        parser_prep.add_argument("--cgo17_runtimes_csv_out", type=str)

        args = parser_prep.parse_args(sys.argv[2:])

        preprocess(**vars(args))


if __name__ == '__main__':
    main()
