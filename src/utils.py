import argparse
import logging
import os
import pathlib
import random
import re
import sys
import warnings
from typing import List, Optional, Union

import numpy as np
import torch
from agent import Setting


def random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup_logging(debug: bool, output: Optional[str] = None) -> None:
    if len(logging.getLogger().handlers) != 0:
        return

    formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] '
                                  '%(message)s (%(module)s.%(funcName)s:%(lineno)s)',
                                  '%Y-%m-%d %H:%M:%S')

    if output is not None:
        handler: logging.Handler = logging.FileHandler(filename=output)
    else:
        handler = logging.StreamHandler(stream=sys.stdout)

    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger('PIL').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    warnings.filterwarnings('ignore', message='Corrupt EXIF data.  Expecting to read 4 bytes but only got 0.')


def get_newest_model_numbers(
    path: Union[str, pathlib.Path],
    size=1,
    max_number=99999,
) -> List[int]:
    regex = re.compile(r'(\d+)_model.pkl')
    numbers = []

    for name in os.listdir(path):
        match = regex.match(name)
        if match and int(match.group(1)) <= max_number:
            numbers.append(int(match.group(1)))
    numbers.sort()
    return numbers[-size:]


def get_newest_record_numbers(
    path: Union[str, pathlib.Path],
    size=1,
    max_number=99999,
) -> List[int]:
    regex = re.compile(r'(\d+)_record.pkl')
    numbers = []
    for name in os.listdir(path):
        match = regex.match(name)
        if match and int(match.group(1)) <= max_number:
            numbers.append(int(match.group(1)))
    numbers.sort()
    return numbers[-size:]


def add_setting_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--timelimit', type=str, default='auto',
        help='Timelimit for searching moves (none, zero, auto or visit count).')
    parser.add_argument(
        '--depthlimit', type=int, default=20,
        help='Depth limit for searching moves (simluation only at more than the specified depth).')
    parser.add_argument(
        '--decision', type=str, default='hybrid', choices=('hybrid', 'value', 'visit'),
        help='Value for move decisions (hybrid, value or visit).')
    parser.add_argument(
        '--collision', type=float, default=0.0,
        help='Penalty of potetial collision moves (ignore potential collisions at 0.0).')
    parser.add_argument(
        '--normalize', type=lambda x: x.strip().lower() in ('y', 'yes', 'on', '1', 'true'),
        default=True, help='Normalize values of each step (yes/no) (default: no).')
    parser.add_argument(
        '--eating', type=float, default=0.0,
        help='Value of eating a food (ignore foods at 0.0).')
    parser.add_argument(
        '--search', type=str, default='ucbm',
        choices=('policy', 'beta', 'ucbm', 'ucbr', 'ucb1', 'puct', 'zero'),
        help='Algorithm of next move selections (default: ucbm).')
    parser.add_argument(
        '--safe', type=float, default=0.0,
        help='Safe parameter at move decisions (most aggresive at 0.0).')
    parser.add_argument(
        '--strict', type=lambda x: x.strip().lower() in ('y', 'yes', 'on', '1', 'true'),
        default=False, help='Evaluate a value of the root node with other edge and leaf nodes (default: no).')
    parser.add_argument(
        '--lookahead', type=int, default=4,
        help='Depth of look-ahead by depth-first search (default: 0).')
    parser.add_argument(
        '--policybase', type=float, default=0.1,
        help='Base of policy (default: 0.1).')
    parser.add_argument(
        '--valuebase', type=float, default=0.0,
        help='Base of value (default: 0.0).')
    parser.add_argument(
        '--valuetail', type=float, default=0.0,
        help='Value at neighbor of own tail (default: 0.0).')
    parser.add_argument(
        '--usepolicy', type=lambda x: x.strip().lower() in ('y', 'yes', 'on', '1', 'true'),
        default=True, help='Use policy model (yes/no) (default: yes).')
    parser.add_argument(
        '--usevalue', type=lambda x: x.strip().lower() in ('y', 'yes', 'on', '1', 'true'),
        default=True, help='Use value model (yes/no) (default: yes).')
    parser.add_argument(
        '--random', type=float, default=0.0,
        help='Probability of random decision (default: 0.0).')


def make_setting(args: argparse.Namespace, **kwargs) -> Setting:
    return Setting(
        timelimit=args.timelimit,
        depthlimit=args.depthlimit,
        decision=args.decision,
        collision=args.collision,
        normalize=args.normalize,
        eating=args.eating,
        search=args.search,
        safe=args.safe,
        strict=args.strict,
        lookahead=args.lookahead,
        policybase=args.policybase,
        valuebase=args.valuebase,
        valuetail=args.valuetail,
        usepolicy=args.usepolicy,
        usevalue=args.usevalue,
        random=args.random,
        **kwargs,
    )
