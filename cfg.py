import argparse
import torch

import logger
import models
import utils

NUM_NODES = {
    'moments': 391,
    'multimoments': 391,
    'kinetics': 608,
}

CRITERIONS = {
    'CE': {'func': torch.nn.CrossEntropyLoss},
    'MSE': {'func': torch.nn.MSELoss},
    'BCE': {'func': torch.nn.BCEWithLogitsLoss},
}

OPTIMIZERS = {
    'SGD': {
        'func': torch.optim.SGD,
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    },
    'Adam': {'func': torch.optim.Adam, 'weight_decay': 5e-4},
}

SCHEDULER_DEFAULTS = {'CosineAnnealingLR': {'T_max': 100}}


METAFILE_FILE = {
    'moments': {
        'train': 'metadata/moments_train_abstraction_sets.json',
        'val': 'metadata/moments_val_abstraction_sets.json',
    },
    'kinetics': {
        'train': 'metadata/kinetics_train_abstraction_sets.json',
        'val': 'metadata/kinetics_val_abstraction_sets.json',
    },
}

FEATURES_FILE = {
    'moments': {
        'train': 'metadata/resnet3d50_moments_train_features.pth',
        'val': 'metadata/resnet3d50_moments_val_features.pth',
        'test': 'metadata/resnet3d50_moments_test_features.pth',
    },
    'kinetics': {
        'train': 'metadata/resnet3d50_kinetics_train_features.pth',
        'val': 'metadata/resnet3d50_kinetics_val_features.pth',
        'test': 'metadata/resnet3d50_kinetics_test_features.pth',
    },
}

EMBEDDING_FILE = {
    'moments': {
        'train': 'metadata/moments_train_embeddings.pth',
        'val': 'metadata/moments_val_embeddings.pth',
    },
    'kinetics': {
        'train': 'metadata/kinetics_train_embeddings.pth',
        'val': 'metadata/kinetics_val_embeddings.pth',
        'test': 'metadata/kinetics_test_embeddings.pth',
    },
}

EMBEDDING_CATEGORIES_FILE = {
    'moments': 'metadata/moments_category_embeddings.pth',
    'kinetics': 'metadata/kinetics_category_embeddings.pth',
}

LIST_FILE = {
    'moments': {
        'train': 'metadata/moments_train_listfile.txt',
        'val': 'metadata/moments_val_listfile.txt',
        'test': 'metadata/moments_test_listfile.txt',
    },
    'kinetics': {
        'train': 'metadata/kinetics_train_listfile.txt',
        'val': 'metadata/kinetics_val_listfile.txt',
        'test': 'metadata/kinetics_test_listfile.txt',
    },
}

RANKING_FILE = {
    'moments': 'metadata/moments_human_abstraction_sets.json',
    'kinetics': 'metadata/kinetics_human_abstraction_sets.json',
}

GRAPH_FILE = {
    'moments': 'metadata/moments_graph.json',
    'kinetics': 'metadata/kinetics_graph.json',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Abstraction Experiments")
    parser.add_argument(
        '-e',
        '--experiment',
        type=str,
        default='AbstractionEmbedding',
        help="name of experiment to run",
    )
    parser.add_argument(
        '-i',
        '--exp_id',
        type=str,
        help="unique name or id of particular experimental run",
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default='moments',
        choices=['moments', 'kinetics'],
        help='name of dataset',
    )

    parser.add_argument(
        '-m',
        '--model_name',
        type=str,
        default='AbstractionEmbeddingModule',
        help='class name of model to instantiate',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=256,
        help='number of elements (sets) in batch',
    )
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--criterion', nargs='+', default=['MSE', 'CE'])
    parser.add_argument('-l', '--loss_weights', nargs='+', default=[1, 1], type=float)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('-s', '--scales', nargs='+', default=[1, 2, 3, 4], type=int)
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--metadata_dir', type=str, default='metadata')
    parser.add_argument('--logger_name', type=str, default='AbstractionLogger')
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_step', type=int, default=None)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--checkpoint_freq', type=int, default=1000)
    parser.add_argument('--cudnn_enabled', default=True, type=utils.str2bool)
    parser.add_argument('--cudnn_benchmark', default=True, type=utils.str2bool)
    parser.add_argument('--clip_gradient', type=int, default=20)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('-bm', '--basemodel_name', type=str, default='resnet3d50')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--return_metric', type=str, default='top1@abstr')
    args = parser.parse_args()
    return args


def get_model(model_name, dataset_name, scales=4, basemodel='resnet3d50'):
    feature_dim = {'resnet3d50': 2048}.get(basemodel, 2048)
    model_dict = {
        'AbstractionEmbeddingModule': {
            'func': models.AbstractionEmbeddingModule,
            'in_features': feature_dim,
            'out_features': feature_dim,
            'num_nodes': NUM_NODES[dataset_name],
            'embedding_dim': 300,
            'bottleneck_dim': 512,
            'scales': scales,
        },
    }.get(model_name)
    model_func = model_dict.pop('func')
    return model_func(**model_dict)


def get_criterion(names=['CE', 'MSE'], cuda=True):
    criterions = {name: CRITERIONS[name]['func']() for name in names}
    if cuda:
        criterions = {name: crit.cuda() for name, crit in criterions.items()}
    return criterions


def get_optimizer(model, optimizer_name, lr=0.001):
    optim_dict = OPTIMIZERS[optimizer_name]
    optim_func = optim_dict.pop('func', torch.optim.Adam)
    optimizer = optim_func(model.parameters(), **{**optim_dict, 'lr': lr})
    return optimizer


def get_scheduler(optimizer, scheduler_name='CosineAnnealingLR', **kwargs):
    sched_func = getattr(torch.optim.lr_scheduler, scheduler_name)
    func_kwargs, _ = utils.split_kwargs_by_func(sched_func, kwargs)
    sched_kwargs = {**SCHEDULER_DEFAULTS.get(scheduler_name, {}), **func_kwargs}
    scheduler = sched_func(optimizer, **sched_kwargs)
    return scheduler


def get_logger(args):
    logger_func = getattr(logger, args.logger_name)
    logger_dict, _ = utils.split_kwargs_by_func(logger_func, vars(args).copy())
    return logger_func(**logger_dict)
