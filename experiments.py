import os
import re
import shutil
import time
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import utils
from utils import cache


class AbstractionEmbedding:

    names = {
        'loss': ['embed', 'abstr'],
        'targets': ['embed', 'abstr'],
        'outputs': ['embed', 'abstr'],
    }

    def __init__(self, **params):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

        self.best_acc1 = 0
        self.check_rootfolders()
        self.load_checkpont()
        self.logger.prepare(self)

        cudnn.enabled = self.params['cudnn_enabled']
        cudnn.benchmark = self.params['cudnn_benchmark']
        self.criterion = {n: c.cuda() for n, c in self.criterion.items()}
        print(f'Starting experiment: {self.name}')

    def run(self):
        if self.params['evaluate']:
            return self.evaluate()

        for epoch in range(self.params['start_epoch'], self.params['num_epochs'],):

            # Train for one epoch
            self.train(epoch)

            # Evaluate on validation set
            if (epoch + 1) % self.val_freq == 0 or epoch == self.num_epochs - 1:
                meters = self.validate(epoch)
                acc1 = meters[self.return_metric].avg

                self.scheduler.step(meters['full'].avg)

                # Remember best acc@1 and save checkpoint
                is_best = acc1 > self.best_acc1
                self.best_acc1 = max(acc1, self.best_acc1)
                self.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        # 'params': self.params,
                        # 'arch': self.model.module.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_acc1': self.best_acc1,
                    },
                    is_best,
                )

    def train(self, epoch):
        # Switch to train mode
        self.model.train()
        # self.meters = self.get_meters(self.__class__.__name__)
        self.meters = self.logger.get_progress_meter(
            epoch, len(self.dataloader['train'])
        )

        end = time.time()
        for i, (input, target) in enumerate(self.dataloader['train']):
            # Measure data loading time
            self.meters['data_time'].update(time.time() - end)

            # Step the experiment
            self.step(input, target)

            # Measure elapsed time
            self.meters['batch_time'].update(time.time() - end)
            end = time.time()

            if i % self.params['log_freq'] == 0:
                self.logger.log(i, mode='train', epoch=epoch)

            if i % self.params['checkpoint_freq'] == 0:
                self.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        # 'params': self.params,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_acc1': self.best_acc1,
                    },
                    False,
                )

            if self.params['max_step'] is not None:
                if i % self.params['max_step'] == 0:
                    break

    def validate(self, epoch, evaluate=False):
        # Switch to evaluate mode
        self.model.eval()
        self.meters = self.logger.get_progress_meter(epoch, len(self.dataloader['val']))

        if evaluate:
            self.probs = defaultdict(list)
            self.preds = defaultdict(list)
            self.outputs = defaultdict(list)
            self.targets = defaultdict(list)

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.dataloader['val']):
                self.meters['data_time'].update(time.time() - end)
                mode = 'eval' if evaluate else 'val'
                # Step the model
                self.step(input, target, mode=mode)

                # Measure elapsed time
                self.meters['batch_time'].update(time.time() - end)
                end = time.time()

                if i % self.params['log_freq'] == 0:
                    self.logger.log(i, mode=mode)

                if self.params['max_step'] is not None:
                    if i % self.params['max_step'] == 0:
                        self.logger.write('Max steps reached!', 'main')
                        break

            if not evaluate:
                msg = self.logger.log_val()
                self.logger.write(msg, 'main')
                self.logger.write(msg, 'val')
            else:
                msg = self.logger.log_eval()
                self.logger.write(msg, 'summary')

        return self.meters

    def step(self, input, target, mode='train'):

        input = self.input_transform(input, mode=mode)
        targets = self.target_transform(target, mode=mode)

        # Compute output => [batch_size, out_size, num_inputs]
        outputs = dict(zip(self.names['outputs'], self.model(input)))
        outputs = self.output_transform(outputs, mode=mode)

        # Compute loss
        loss = {
            name: self.loss_weights[name]
            * self.criterion[name](outputs[name], targets[name])
            for name in self.names['outputs']
        }
        loss['full'] = sum(loss.values())
        for name, value in loss.items():
            self.meters[name].update(value.item(), self.batch_size)

        # Measure metrics
        acc1, acc5 = utils.accuracy(outputs['abstr'], targets['abstr'], topk=(1, 5))

        self.meters['top1@abstr'].update(acc1.item(), self.batch_size)
        self.meters['top5@abstr'].update(acc5.item(), self.batch_size)

        inds = {
            1: (0, 4),
            2: (4, 10),
            3: (10, 14),
            4: (14, 15),
        }
        inds = {k: v for k, v in inds.items() if k >= min(self.scales)}
        for scale, (start_idx, stop_idx) in inds.items():
            acc1, acc5 = utils.accuracy(
                outputs['abstr'][..., start_idx:stop_idx],
                targets['abstr'][..., start_idx:stop_idx],
                topk=(1, 5),
            )
            self.meters[f'top1@abstr_{scale}'].update(acc1.item(), self.batch_size)
            self.meters[f'top5@abstr_{scale}'].update(acc5.item(), self.batch_size)
        if mode == 'train':
            # Compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss['full'].backward()

            # Clip gradients
            if self.params['clip_gradient'] is not None:
                clip_gradient = self.params['clip_gradient']
                total_norm = clip_grad_norm_(self.model.parameters(), clip_gradient)
                if total_norm > clip_gradient:
                    print(
                        f'clipping gradient: {total_norm:.4f} with coef {(clip_gradient/total_norm):.4f}'
                    )

            # Update weights
            self.optimizer.step()
        elif mode == 'eval':
            for name in self.names['outputs']:
                probs, preds = F.softmax(outputs[name], 1).sort(1, True)
                self.probs[name].append(probs.detach().cpu())
                self.preds[name].append(preds.detach().cpu())
                self.targets[name].append(targets[name].detach().cpu())
                self.outputs[name].append(outputs[name].detach().cpu())

    def target_transform(self, target, mode='train'):
        targets = {}
        min_scale = min(self.scales)
        offset = {1: 0, 2: 4, 3: 10, 4: 15}.get(min_scale)
        for name, tgt in zip(self.names['targets'], target):
            targets[name] = tgt.cuda(non_blocking=True)[:, offset:]
            self.batch_size = tgt.size(0)
        return targets

    def input_transform(self, input, mode='train'):
        return input

    def output_transform(self, output, mode='train'):
        return output

    @cache
    def name(self):
        name = '_'.join(
            map(
                str,
                [
                    self.__class__.__name__,
                    self.exp_id,
                    self.params['dataset_name'],
                    self.params['basemodel_name'],
                    '-'.join(map(str, self.param_names['loss_weights'])),
                    '-'.join(map(str, self.param_names['criterion'])),
                    '-'.join(map(str, [self.param_names['optimizer'], self.lr])),
                    self._model.name,
                ],
            )
        )
        name = self.params['resume'] or name
        name = re.sub(r'_(checkpoint|best).pth.tar$', '', name)
        name = self.params['prefix'] + name.split('/')[-1]
        name = type(self).__name__ + '_' + '_'.join(name.split('_')[1:])
        return name

    def check_rootfolders(self):
        """Create log and model folder."""
        folders_util = [
            self.params['log_dir'],
            self.params['output_dir'],
            self.params['metadata_dir'],
            self.params['checkpoint_dir'],
        ]
        for folder in folders_util:
            os.makedirs(folder, exist_ok=True)

    def save_name(self, save_type='EVAL', mode='val', format='torch'):
        ext = {'torch': '.pth', 'pickle': '.pkl', 'npz': '.npz'}.get(format, '')
        name = '_'.join(
            map(
                str,
                [
                    save_type.upper(),
                    mode.upper(),
                    '-'.join(self.attrs),
                    '-'.join(map(str, self.set_maxmin)),
                    self.name,
                ],
            )
        )
        return self.params['prefix'] + name + ext

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', freq=5):
        checkpoint_dir = os.path.join(
            self.params['checkpoint_dir'],
            self.__class__.__name__,
            '_'.join([type(self._model).__name__]),
        )
        #   type(self._model.model).__name__]))
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(
            checkpoint_dir, f'{self.name}_checkpoint.pth.tar'
        )
        best_file = checkpoint_file.replace('checkpoint.pth.tar', 'best.pth.tar')
        epoch_file = checkpoint_file.replace(
            'checkpoint.pth.tar', f'epoch_{state["epoch"]}.pth.tar'
        )
        # torch.save(state, checkpoint_file, pickle_protocol=4)
        torch.save(state, checkpoint_file)
        if is_best:
            shutil.copyfile(checkpoint_file, best_file)
        elif state['epoch'] % freq == 0:
            shutil.copyfile(checkpoint_file, epoch_file)

    def load_checkpont(self):
        if self.params['resume'] is None:
            self.params['checkpoint'] = None
            return
        file = self.params['resume']
        if os.path.exists(file):
            print(("=> loading checkpoint '{}'".format(file)))
            checkpoint = torch.load(file)
            self.params['start_epoch'] = checkpoint['epoch']
            self.best_acc1 = checkpoint['best_acc1']
            self.model.load_state_dict(checkpoint['state_dict'])
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except (KeyError, AttributeError):
                pass
            else:
                print(
                    (
                        "=> loaded checkpoint '{}' (epoch {})".format(
                            file, checkpoint['epoch']
                        )
                    )
                )
                print(f'Best Acc@1: {self.best_acc1:.3f}')
                torch.cuda.empty_cache()
        else:
            print(("=> no checkpoint found at '{}'".format(file)))

    @cache
    def save_prefix(self):
        return os.path.join(
            self.__class__.__name__,
            '_'.join([type(self._model).__name__, type(self._model.model).__name__]),
        )
