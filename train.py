import torch.nn as nn
from torch.utils import data

import cfg
import datasets
import experiments as exp

if __name__ == '__main__':
    args = cfg.parse_args()

    exp_func = getattr(exp, args.experiment)

    # Model
    _model = cfg.get_model(
        args.model_name, args.dataset, scales=args.scales, basemodel=args.basemodel_name
    )
    model = nn.DataParallel(_model)
    model = model.cuda()

    # Optimizer and Scheduler
    optimizer = cfg.get_optimizer(model, args.optimizer, lr=args.lr)
    scheduler = cfg.get_scheduler(optimizer, T_max=args.num_epochs)

    # Criterion
    criterion_func = cfg.get_criterion(args.criterion, cuda=True)
    criterion = {'embed': criterion_func['MSE'], 'abstr': criterion_func['CE']}
    loss_weights = {
        name: lw for name, lw in zip(exp_func.names['loss'], args.loss_weights)
    }

    # Dataloading
    train_dataset = datasets.get_cached_abstraction_dataset(
        args.dataset, 'train', basemodel=args.basemodel_name
    )
    val_dataset = datasets.get_cached_abstraction_dataset(
        args.dataset, 'val', basemodel=args.basemodel_name
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True
    )

    dataset = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {'train': train_loader, 'val': val_loader}

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
    }

    runner = exp_func(**params)
    runner.run()
