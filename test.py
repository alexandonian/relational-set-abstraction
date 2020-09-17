import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils import data

import cfg
import datasets
import experiments as exp
import logger
import utils

args = cfg.parse_args()

exp_func = getattr(exp, args.experiment)

# Model
_model = cfg.get_model(
    args.model_name, args.dataset, scales=args.scales, basemodel=args.basemodel_name
)
model = nn.DataParallel(_model)
model = model.cuda()

# Optimizer
optimizer = cfg.get_optimizer(model, args.optimizer, lr=args.lr)
scheduler = cfg.get_scheduler(optimizer)

# Criterion
criterion_func = cfg.get_criterion(args.criterion, cuda=True)
criterion = {'embed': criterion_func['MSE'], 'abstr': criterion_func['CE']}
loss_weights = {name: lw for name, lw in zip(exp_func.names['loss'], args.loss_weights)}

# Dataloading
val_dataset = datasets.get_cached_abstraction_dataset(
    args.dataset, 'val', basemodel=args.basemodel_name
)
test_dataset = datasets.get_cached_ranking_dataset(
    args.dataset, 'test', basemodel=args.basemodel_name
)
outlier_dataset = datasets.get_cached_outlier_dataset(
    args.dataset, 'test', basemodel=args.basemodel_name
)
val_loader = data.DataLoader(
    val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True
)

dataset = {'val': val_dataset, 'test': test_dataset}

dataloaders = {'val': val_loader}

param_names = {
    'criterion': args.criterion,
    'optimizer': args.optimizer,
    'loss_weights': args.loss_weights,
}

log = cfg.get_logger(args)

params = {
    **vars(args),
    'dataset_name': args.dataset,
    'param_names': param_names,
    'logger': log,
    'model': model,
    '_model': _model,
    'optimizer': optimizer,
    'loss_weights': loss_weights,
    'scheduler': scheduler,
    'dataset': dataset,
    'dataloader': dataloaders,
    'criterion': criterion,
    'evaluate': True,
}


def eval_ranking(runner, test_dataset):
    runner._model.eval()
    runner.model.eval()

    rankings = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            try:
                ref_features, rank_features = test_dataset[i]
            except IndexError:
                continue

            ref_features = ref_features.unsqueeze(0).cuda()
            rank_features = rank_features.unsqueeze(0).cuda()
            ranking = runner._model.rank((ref_features, rank_features)).cpu()

            rankings.append(ranking)

            if i % 1000 == 0:
                print(f'[{i} / {len(test_dataset)}]')

    rankings = torch.cat(rankings).numpy()

    gt_rankings = test_dataset.get_dists()
    set_splits = {
        size: len(d) for size, d in test_dataset.record_set.records_dict.items()
    }

    curr_idx = 0
    corrs = {}
    for set_size, num_videos in set_splits.items():

        gtrnk = gt_rankings[curr_idx : curr_idx + num_videos]
        rnk = rankings[curr_idx : curr_idx + num_videos]

        curr_corr = np.mean(
            [stats.spearmanr(g, r).correlation for g, r in zip(gtrnk, rnk)]
        )

        corrs[set_size] = curr_corr

        msg = f'Avg Rank Correlation (N={set_size}): {curr_corr:0.4f}'
        runner.logger.write(msg, 'summary')

        curr_idx += num_videos

    avg_corr = np.mean(
        [stats.spearmanr(g, r).correlation for g, r in zip(gt_rankings, rankings)]
    )

    msg = f'Overall Avg Rank Correlation: {avg_corr}'
    runner.logger.write(msg, 'summary')


def eval_outlier(runner, outlier_dataset):
    meters = {
        3: {
            'top1': logger.AverageMeter('outlr3'),
            'top2': logger.AverageMeter('outlr3'),
        },
        4: {
            'top1': logger.AverageMeter('outlr4'),
            'top2': logger.AverageMeter('outlr4'),
        },
        5: {
            'top1': logger.AverageMeter('outlr5'),
            'top2': logger.AverageMeter('outlr5'),
        },
    }
    runner._model.eval()
    runner.model.eval()
    records = []
    dists = []
    outliers = []
    with torch.no_grad():
        for i in range(len(outlier_dataset)):
            try:
                set_features, outlr_target = outlier_dataset[i]
                record = outlier_dataset.record_set[i]

            except IndexError as e:
                print(e)
                continue
            num_inputs = set_features.size(0)
            set_features = set_features.unsqueeze(0).cuda()
            outlr_target = outlr_target.cuda()

            dist, outlr_out = runner._model.find_outlier(set_features)

            records.append(record)
            outliers.append(outlr_out.tolist())
            dists.append(dist.tolist())

            # acc1, acc2 = utils.accuracy(outlr_out, outlr_target, topk=(1, 2))
            acc1, acc2 = utils.accuracy(dist, outlr_target, topk=(1, 2))

            meters[num_inputs]['top1'].update(acc1.item(), 1)
            meters[num_inputs]['top2'].update(acc2.item(), 1)

            if i % 100 == 0:
                print(
                    f'[{i} / {len(outlier_dataset)}]\t'
                    f'3Acc@1 {meters[3]["top1"].val:.3f} ({meters[3]["top1"].avg:.3f})\t'
                    f'3Acc@2 {meters[3]["top2"].val:.3f} ({meters[3]["top2"].avg:.3f})\t'
                    f'4Acc@1 {meters[4]["top1"].val:.3f} ({meters[4]["top1"].avg:.3f})\t'
                    f'4Acc@2 {meters[4]["top2"].val:.3f} ({meters[4]["top2"].avg:.3f})\t'
                    f'5Acc@1 {meters[5]["top1"].val:.3f} ({meters[5]["top1"].avg:.3f})\t'
                    f'5Acc@2 {meters[5]["top2"].val:.3f} ({meters[5]["top2"].avg:.3f})'
                )

        msg = (
            f'N = 3 Acc@1 ({meters[3]["top1"].avg:.3f}) Acc@2  ({meters[3]["top2"].avg:.3f})\n'
            f'N = 4 Acc@1 ({meters[4]["top1"].avg:.3f}) Acc@2  ({meters[4]["top2"].avg:.3f})\n'
            f'N = 5 Acc@1 ({meters[5]["top1"].avg:.3f}) Acc@2  ({meters[5]["top2"].avg:.3f})\n'
        )
        runner.logger.write(msg, 'summary')

    return records, dists, outliers


def eval_abstraction(runner):
    runner.validate(epoch=None, evaluate=True)
    return runner


if __name__ == '__main__':

    runner = exp_func(**params)
    eval_abstraction(runner)
    eval_ranking(runner, test_dataset)
    eval_outlier(runner, outlier_dataset)
