import os
from collections import defaultdict

from utils import HTML


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", default_fmt=':.3f'):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.default_fmt = default_fmt

    def display(self, batch, verbose=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        m = list(self.meters.values()) if isinstance(self.meters, dict) else self.meters
        entries += [str(meter) for meter in m]
        msg = '\t'.join(entries)
        print(msg) if verbose else None
        return msg

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def reset(self):
        m = list(self.meters.values()) if isinstance(self.meters, dict) else self.meters
        for meter in m:
            meter.reset()

    def __getitem__(self, key):
        if key not in self.meters:
            self.meters[key] = AverageMeter(key, self.default_fmt)
        return self.meters[key]


class Logger(object):

    parent_names = ['name', 'optimizer']

    def __init__(self, log_dir='logs', output_dir='outputs', **kwargs):
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.kwargs = kwargs

    def prepare(self, parent):
        self.parent = parent
        self.modes = list(parent.dataloader.keys())
        self.parent_cls_name = os.path.join(
            type(parent).__name__, '_'.join([type(parent._model).__name__])
        )
        self.exp_log_dir = os.path.join(self.log_dir, self.parent_cls_name)
        self.output_dir = os.path.join(self.output_dir, self.parent_cls_name)
        self.dlen = {mode: len(self.parent.dataloader[mode]) for mode in self.modes}
        self.params = {**self.kwargs, **self.parent.params}
        for name in self.parent_names:
            setattr(self, name, getattr(self.parent, name))

        self.logs = {}
        for d in [self.exp_log_dir, self.output_dir]:
            os.makedirs(d, exist_ok=True)
        if self.parent.params['resume'] is None:
            self.logs['main'] = open(
                os.path.join(self.exp_log_dir, f'{self.name}.txt'), 'w'
            )
            self.logs['val'] = open(
                os.path.join(self.exp_log_dir, f'{self.name}_val.txt'), 'w'
            )
        else:
            self.logs['main'] = open(
                os.path.join(self.exp_log_dir, f'{self.name}.txt'), 'a'
            )
            self.logs['val'] = open(
                os.path.join(self.exp_log_dir, f'{self.name}_val.txt'), 'a'
            )
        if self.parent.params['evaluate']:
            self.logs['eval'] = open(
                os.path.join(self.exp_log_dir, f'{self.name}_eval.txt'), 'w'
            )
            self.logs['summary'] = open(
                os.path.join(self.exp_log_dir, f'{self.name}_summary.txt'), 'w'
            )

    def open_file(self, filename):
        """Open file for plotting."""
        try:
            self.file.close()
        except Exception:
            pass
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, 'w')
        self.file.write(HTML.head())

    def write(self, out, name='main', verbose=True):
        print(out) if verbose else None
        out += '\n' if not out.endswith('\n') else ''
        self.logs[name].write(out)
        self.logs[name].flush()

    def log(self, step, mode='train', epoch=None):
        if mode == 'train':
            out = (
                'Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch,
                    step,
                    self.dlen['train'],
                    batch_time=self.parent.batch_time,
                    data_time=self.parent.data_time,
                    loss=self.parent.losses,
                    top1=self.parent.top1,
                    top5=self.parent.top5,
                    lr=self.parent.optimizer.param_groups[-1]['lr'],
                )
            )
            self.write(out, 'main')
        elif mode == 'val':
            out = (
                'Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    step,
                    self.dlen['val'],
                    loss=self.parent.losses,
                    batch_time=self.parent.batch_time,
                    top1=self.parent.top1,
                    top5=self.parent.top5,
                )
            )
            self.write(out, 'main', verbose=False)
            self.write(out, 'val')

    @property
    def meters(self):
        return self.parent.meters


class AbstractionLogger(Logger):
    def get_progress_meter(self, epoch, data_len):
        meters = defaultdict(lambda k: AverageMeter(k, ':.3f'))
        meters['batch_time'] = AverageMeter('Time', ':.3f')
        meters['data_time'] = AverageMeter('Data', ':.3f')
        for k in ['abstr', 'embed', 'full']:
            meters[k] = AverageMeter(k, ':.4f')
        progress = ProgressMeter(data_len, meters, prefix="Epoch: [{}]".format(epoch))
        return progress

    def log(self, step, mode='train', epoch=None, no_log=False):
        if mode == 'train':
            out = self.meters.display(step, verbose=False)
            if not no_log:
                self.write(out, 'main')
            else:
                print(out)
        else:
            out = self.meters.display(step, verbose=False)
            if not no_log:
                self.write(out, mode)
                if mode != 'eval':
                    self.write(out, 'main', verbose=False)
            else:
                print(out)

    def log_val(self):
        msg = 'Testing Results: Loss: '
        msg += self.meters.display(0)
        return_metric = self.params['return_metric']
        acc1 = self.meters[return_metric].avg
        best = max(acc1, self.parent.best_acc1)
        msg += f'\nBest {return_metric}: {best:.3f}\n'
        return msg

    def log_eval(self):
        msg = '-----Evaluation is finished------\n'
        metric_names = [n for n in self.meters.meters if n.startswith('top')]
        for name in metric_names:
            msg += f'{self.meters[name]}\n'
        return msg
