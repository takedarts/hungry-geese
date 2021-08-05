'''This script trains a CNN model by using the newest game records.
This script saves a model as 2 files:
  weights file (*.pth): the model weights and the training logs are saved.
  traced file (*.pkl): the model weights and the model structure are saved.
The traced model file is created by using torch.jit.trace and torch.jit.save.
'''
import argparse
import bisect
import logging
import pathlib
import pickle
from typing import Any, Dict, Iterator, List, Tuple
import itertools
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from kaggle_environments.envs.hungry_geese.hungry_geese \
    import Configuration, Observation

from agent import Setting, World
from config import INPUT_AREA_SIZE, INPUT_CHANNELS
from model import DummyModel, NormalModel, STEM_TYPE, HEAD_TYPE
from record import convert_records2steps
from utils import get_newest_record_numbers, setup_logging

parser = argparse.ArgumentParser(
    description='Traning a model.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--records', type=int, default=60, help='Number of records.')
parser.add_argument('--iter', type=int, default=800_000, help='Number of iterations.')
parser.add_argument('--batch', type=int, default=256, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--decay', type=float, default=0.00001, help='Weight decay.')
parser.add_argument('--workers', type=int, default=1, help='Number of workers.')
parser.add_argument('--gpu', type=int, default=None, help='GPU ID.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')

LOGGER = logging.getLogger(__name__)
DUMMY_MODEL = DummyModel()


class Datum(object):
    '''Create input features from a game record.
    This class manages a game record, and creates input features of the specified steps.
    '''

    def __init__(
        self,
        config: Dict[str, Any],
        records: List[List[List[int]]],
        result: List[int],
    ) -> None:
        self.config = config
        self.records = records
        self.result = result
        self.winner = result.index(0)
        self.lasts = [len(self.records)] * 4

        # search last steps
        for sidx in range(len(self.records) - 1, -1, -1):
            for pidx in range(4):
                if len(self.records[sidx][pidx]) == 0:
                    self.lasts[pidx] = sidx

    def __len__(self) -> int:
        return sum(self.lasts) - 4

    def __getitem__(
        self,
        index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pidx = 0
        sidx = index

        while sidx >= self.lasts[pidx] - 1:
            sidx -= self.lasts[pidx] - 1
            pidx += 1

        # make world
        steps = convert_records2steps(self.records[sidx:sidx + 2])
        steps[1][0]['observation']['food'] = [
            v for v in steps[1][0]['observation']['food'] if random.random() > 0.5]
        steps[1][0]['observation']['step'] = sidx + 1

        world = World(DUMMY_MODEL, Configuration(self.config), Setting())
        world.apply(
            Observation(steps[1][0]['observation']),
            use_model=False, update_move=False,
            prevobs=Observation(steps[0][0]['observation']))

        # policy
        directions = [(world.directions[pidx] - 1 + i) % 4 for i in range(3)]
        action = self.records[sidx + 1][5][pidx]
        policy = np.array([directions[i] == action for i in range(3)], dtype=np.float32)

        # value
        value = np.zeros([3], dtype=np.float32)
        value[self.result[pidx]:] = 1.0

        alives = len([g for g in steps[1][0]['observation']['geese'] if len(g) != 0])
        value_mask = np.zeros([3], dtype=np.float32)
        value_mask[:alives - 1] = 1.0

        # augmentation
        if random.random() > 0.5:
            indata = np.array(world.input[pidx][:, :, ::-1])
            policy[:] = [policy[2], policy[1], policy[0]]
        else:
            indata = np.array(world.input[pidx])

        return indata, policy, value, value_mask


class Dataset(torch.utils.data.Dataset):
    '''Dataset class for training.'''

    def __init__(self, data: List[Datum]) -> None:
        self.data = data
        self.offset = [0]

        for datum in self.data:
            self.offset.append(self.offset[-1] + len(datum))

    def __len__(self) -> int:
        return self.offset[-1]

    def __getitem__(
        self,
        index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        number = bisect.bisect_right(self.offset, index) - 1
        offset = self.offset[number]
        return self.data[number][index - offset]


def repeat_loader(data_loader: torch.utils.data.DataLoader) -> Iterator[Any]:
    '''Wrapper for making repeatable data loader.'''
    for loader in itertools.repeat(data_loader):
        for v in loader:
            yield v


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.debug)

    data_path = pathlib.Path(__file__).parent.parent / 'data'
    record_numbers = get_newest_record_numbers(data_path, size=args.records)
    model_path = data_path / f'{record_numbers[-1]:05d}_model.pth'
    trace_path = data_path / f'{record_numbers[-1]:05d}_model.pkl'

    # make model
    model = NormalModel()

    LOGGER.info(
        'start training: %s (stem=%d, head=%d, blocks=%d)',
        trace_path, STEM_TYPE, HEAD_TYPE, len(model.blocks))

    # make data loader
    data = []

    for num in record_numbers:
        record_path = data_path / f'{num:05d}_record.pkl'
        LOGGER.info('load records: %s', record_path)
        with open(record_path, 'rb') as reader:
            data.extend([Datum(c, r, v) for c, r, v in pickle.load(reader)])

    loader = repeat_loader(torch.utils.data.DataLoader(
        Dataset(data), batch_size=args.batch,
        shuffle=True, drop_last=True, num_workers=args.workers))

    # make optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [round(args.iter * i / 4) for i in range(1, 4)], gamma=0.1)

    if args.gpu is not None:
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

        y = model(x)
        py = y[:, :3]
        vy = y[:, 3:]

        pe = nn.functional.mse_loss(py, pt)
        ve = nn.functional.mse_loss(vy, vt, reduction='none')
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
    LOGGER.info('save a snapshot: %s', model_path)
    torch.save({'model': model.state_dict(), 'log': logs}, model_path)

    # save traced model
    inputs = torch.randn([1, INPUT_CHANNELS, INPUT_AREA_SIZE, INPUT_AREA_SIZE], dtype=torch.float32)
    model = model.cpu()
    model.eval()
    model = torch.jit.trace(model, inputs)

    LOGGER.info('save a traced model: %s', trace_path)
    torch.jit.save(model, str(trace_path))


if __name__ == '__main__':
    main()
