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

"""Launcher for experiments on a SLURM cluster."""

import argparse
import os


CONFIG_DIR = ''
TC_CONFIGS = {
    'magni': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'deeptune': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_ast': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_ast_astonly': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_llvm': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_llvm_cfgonly': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_llvm_cfgdataflowonly': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_llvm_cfgdataflowcallonly': {
        'slurm': {
            'config': 'ml.slurm'
        }
    }
}
DEVMAP_CONFIGS = {
    'random': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'static': {
        'slurm': {
            'config': 'ml.slurm',
        }
    },
    'grewe': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'deeptune': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_ast': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_ast_astonly': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_llvm': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_llvm_cfgonly': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_llvm_cfgdataflowonly': {
        'slurm': {
            'config': 'ml.slurm'
        }
    },
    'gnn_llvm_cfgdataflowcallonly': {
        'slurm': {
            'config': 'ml.slurm'
        }
    }
}


# ThreadCoarsening experiment
def build_tc_experiment_args(method, report_write_dir, seed):
    experiment_arg = ['experiment']
    dataset_args = ['--runtimes_csv data/tc/pact-2014-runtimes.csv',
                    '--oracles_csv data/tc/pact-2014-oracles.csv',
                    '--devmap_amd_csv data/tc/cgo17-amd.csv']

    args = experiment_arg \
            + dataset_args \
            + ['--' + method] \
            + ['--report_write_dir', report_write_dir] \
            + ['--seed', str(seed)] \

    return args


def build_tc_experiment_infos(report_write_root_dir, num_iterations, methods):
    cmds = []

    if len(methods) == 0:
        methods = ['gnn_ast', 'gnn_ast_astonly', 'gnn_llvm', 'gnn_llvm_cfgonly', 'gnn_llvm_cfgdataflowonly', 'gnn_llvm_cfgdataflowcallonly', 'deeptune', 'magni']

    for method in methods:
        for seed in range(1, num_iterations + 1):
            # Create report dir
            report_write_dir = os.path.join(report_write_root_dir, 'tc_%s' % (method))
            if not os.path.exists(report_write_dir):
                os.makedirs(report_write_dir)

            # Build command
            cmd = ['lib/experiment/ThreadCoarsenExperiment.py']
            cmd += build_tc_experiment_args(method, report_write_dir, seed)

            cmds.append({
                'cmd': cmd,
                'config': TC_CONFIGS[method]
            })

    return cmds


# DevMap experiment
def build_devmap_experiment_args(method, fold_mode, report_write_dir, seed):
    experiment_arg = ['experiment']
    dataset_args = ['--dataset_nvidia data/devmap/nvidia.csv',
                    '--dataset_amd data/devmap/amd.csv']

    args = experiment_arg \
            + dataset_args \
            + ['--' + method] \
            + ['--fold_mode', fold_mode] \
            + ['--report_write_dir', report_write_dir] \
            + ['--seed', str(seed)] \

    return args


def build_devmap_experiment_infos(report_write_root_dir, num_iterations, methods):
    cmds = []

    if len(methods) == 0:
        methods = ['gnn_ast', 'gnn_ast_astonly', 'gnn_llvm', 'gnn_llvm_cfgonly', 'gnn_llvm_cfgdataflowonly', 'gnn_llvm_cfgdataflowcallonly', 'deeptune', 'random', 'static', 'grewe']

    for method in methods:
        for fold_mode in ['random', 'grouped']:
            for seed in range(1, num_iterations + 1):
                # Create report dir
                report_write_dir = os.path.join(report_write_root_dir, 'devmap_%s_%s' % (method, fold_mode))
                if not os.path.exists(report_write_dir):
                    os.makedirs(report_write_dir)

                # Build command
                cmd = ['lib/experiment/DevMapExperiment.py']
                cmd += build_devmap_experiment_args(method, fold_mode, report_write_dir, seed)

                cmds.append({
                    'cmd': cmd,
                    'config': DEVMAP_CONFIGS[method]
                })

    return cmds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment')
    parser.add_argument('--num_iterations')
    parser.add_argument('--max_threads')
    parser.add_argument('--report_write_root_dir')
    parser.add_argument('--methods', '--names-list', nargs='+', default=[])

    args = parser.parse_args()

    # Build job list
    if args.experiment == 'devmap':
        infos = build_devmap_experiment_infos(
            os.path.join(args.report_write_root_dir, 'devmap'),
            int(args.num_iterations),
            args.methods)
    elif args.experiment == 'tc':
        infos = build_tc_experiment_infos(
            os.path.join(args.report_write_root_dir, 'tc'),
            int(args.num_iterations),
            args.methods)

    print('Number of jobs: %i' % len(infos))

    # Execute jobs
    for info in infos:
        # Build complete command
        complete_command = ' '.join(['sbatch'] + [os.path.join(CONFIG_DIR, info['config']['slurm']['config'])] +
                                    ['\"'] +
                                    ['python'] + info['cmd'] +
                                    ['\"'])
        print(complete_command)
        os.system(complete_command)

if __name__ == '__main__':
    main()
