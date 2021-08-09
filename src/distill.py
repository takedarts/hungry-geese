'''This script trains a distilled CNN model by using the newest game records.
This script saves a model as 2 files:
  weights file (*.pth): the model weights and the training logs are saved.
  traced file (*.pkl): the model weights and the model structure are saved.
The traced model file is created by using torch.jit.trace and torch.jit.save.
The distillation script is basically same as the training script (train.py).
'''
import argparse
import logging
import pathlib
import pickle

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from config import INPUT_AREA_SIZE, INPUT_CHANNELS
from model import Model, NormalModel, SmallModel, TinyModel, STEM_TYPE, HEAD_TYPE
from utils import get_newest_model_numbers, get_newest_record_numbers, setup_logging
from train import Datum, Dataset, repeat_loader

parser = argparse.ArgumentParser(
    description='Traning a distilled model.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '--type', type=str, default='normal', choices=('normal', 'small', 'tiny'),
    help='Type of trained model (normal, small or tiny, default:normal).')
parser.add_argument('--records', type=int, default=60, help='Number of records.')
parser.add_argument(
    '--base', type=lambda x: x.split(','), default=[],
    help='Base models (if not specified, the newest 3 models are selected).')
parser.add_argument('--iter', type=int, default=800_000, help='Number of iterations.')
parser.add_argument('--batch', type=int, default=256, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--decay', type=float, default=0.00001, help='Weight decay.')
parser.add_argument('--workers', type=int, default=1, help='Number of workers.')
parser.add_argument('--gpu', type=int, default=None, help='GPU ID.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')

LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.debug)

    data_path = pathlib.Path(__file__).parent.parent / 'data'
    record_numbers = get_newest_record_numbers(data_path, size=args.records)
    model_path = data_path / f'{record_numbers[-1]:05d}_{args.type}.pth'
    trace_path = data_path / f'{record_numbers[-1]:05d}_{args.type}.pkl'

    # model
    if args.type == 'normal':
        model: Model = NormalModel()
    elif args.type == 'small':
        model = SmallModel()
    else:
        model = TinyModel()

    LOGGER.info(
        'start training: %s (stem=%d, head=%d, blocks=%d)',
        trace_path, STEM_TYPE, HEAD_TYPE, len(model.blocks))

    # base model
    base_models = []

    if len(args.base) == 0:
        numbers = get_newest_model_numbers(data_path, size=3)
        args.base = [data_path / f'{n:05d}_model.pkl' for n in numbers]

    for base_path in args.base:
        LOGGER.info('load base model: %s', base_path)
        base_models.append(torch.jit.load(str(base_path)))

    # load data
    data = []

    for num in record_numbers:
        record_path = data_path / f'{num:05d}_record.pkl'
        LOGGER.info('load records: %s', record_path)
        with open(record_path, 'rb') as record_reader:
            data.extend([Datum(c, r, v) for c, r, v in pickle.load(record_reader)])

    loader = repeat_loader(torch.utils.data.DataLoader(
        Dataset(data), batch_size=args.batch,
        shuffle=True, drop_last=True, num_workers=args.workers))

    # optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [round(args.iter * i / 4) for i in range(1, 4)], gamma=0.1)

    if args.gpu is not None:
        base_models = [m.cuda(device=args.gpu) for m in base_models]
        model = model.cuda(device=args.gpu)

    # train
    iteration = 0
    policy_err_total = 0.0
    policy_acc_total = 0.0
    value_err_total = 0.0
    value_acc_total = 0.0
    data_count = 0
    logs = []

    model.train()

    for x, pt, vt, vm in loader:
        if args.gpu is not None:
            x = x.cuda(device=args.gpu)
            pt = pt.cuda(device=args.gpu)
            vt = vt.cuda(device=args.gpu)
            vm = vm.cuda(device=args.gpu)

        with torch.no_grad():
            z = torch.stack([m(x) for m in base_models], dim=0).mean(dim=0)
            pz = z[:, :3]
            vz = z[:, 3:]

        y = model(x)
        py = y[:, :3]
        vy = y[:, 3:]

        pe = nn.functional.mse_loss(py, (pt + pz) * 0.5)
        ve = nn.functional.mse_loss(vy, (vt + vz) * 0.5, reduction='none')
        ve = ((ve * vm).sum(dim=1) / (vm.sum(dim=1) + 1e-6)).mean()

        optimizer.zero_grad()
        (pe + ve).backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            policy_acc = (py.argmax(dim=1) == pt.argmax(dim=1)).float().mean()
            value_acc = ((vy * vm > 0.5) == (vt * vm > 0.5)).float().min(dim=1)[0].mean()

        iteration += 1
        policy_err_total += float(pe.detach().cpu())
        policy_acc_total += float(policy_acc.detach().cpu())
        value_err_total += float(ve.detach().cpu())
        value_acc_total += float(value_acc.detach().cpu())
        data_count += 1

        if args.debug or iteration % max(args.iter // 1000, 1) == 0 or iteration == args.iter:
            logs.append({
                'iteration': iteration,
                'policy_loss': policy_err_total / data_count,
                'policy_accuracy': policy_acc_total / data_count,
                'value_loss': value_err_total / data_count,
                'value_accuracy': value_acc_total / data_count,
                'lr': float(optimizer.param_groups[0]['lr']),
            })
            policy_err_total = 0.0
            policy_acc_total = 0.0
            value_err_total = 0.0
            value_acc_total = 0.0
            data_count = 0
            LOGGER.info(
                '[%(iteration)d] '
                'policy(loss/acc)=%(policy_loss)6f/%(policy_accuracy).4f, '
                'value(loss/acc)=%(value_loss).6f/%(value_accuracy).4f, '
                'lr=%(lr)6f',
                logs[-1])

        if iteration >= args.iter:
            break

    # save
    LOGGER.info('save model snapshot: %s', model_path)
    torch.save({'model': model.state_dict(), 'log': logs}, model_path)

    # save traced model
    inputs = torch.randn([1, INPUT_CHANNELS, INPUT_AREA_SIZE, INPUT_AREA_SIZE], dtype=torch.float32)
    model = model.cpu()
    model.eval()
    model = torch.jit.trace(model, inputs)

    LOGGER.info('save traced model: %s', trace_path)
    torch.jit.save(model, str(trace_path))


if __name__ == '__main__':
    main()
