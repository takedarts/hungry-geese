import argparse
import json
import logging
import multiprocessing.managers
import pathlib
import pickle
import queue
import time
from typing import Any, Callable, Dict, List, Tuple

import torch.jit
from kaggle_environments import make, Environment
from kaggle_environments.envs.hungry_geese.hungry_geese \
    import Action, Configuration, Observation

from agent import Agent
from model import DummyModel
from record import convert_state2result, convert_steps2records
from utils import (add_setting_arguments, get_newest_model_numbers,
                   get_newest_record_numbers, make_setting, random_seed,
                   setup_logging)

parser = argparse.ArgumentParser(
    description='Generate training data.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--games', type=int, default=1000, help='Number of games generated.')
parser.add_argument('--workers', type=int, default=1, help='Number of workers.')
add_setting_arguments(parser)
parser.set_defaults(timelimit='320')
parser.set_defaults(collision=0.05)
parser.set_defaults(eating=0.001)
parser.set_defaults(search='ucbr')
parser.set_defaults(lookahead=3)
parser.set_defaults(valuebase=0.1)
parser.set_defaults(policybase=0.1)
parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))), default=[], help='GPU IDs.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')

LOGGER = logging.getLogger(__name__)
TIMEOUT = 5.0


def make_agent(
    model_path: pathlib.Path,
    timestamp: multiprocessing.managers.ValueProxy,
    env: Environment,
    args: argparse.Namespace,
) -> Callable:
    def check_timeout(agent: Agent, obs: Observation, action: Action) -> None:
        if time.time() - timestamp.value > TIMEOUT:
            raise Exception('timeout')

    # model
    if model_path.is_file():
        model = torch.jit.load(model_path)
    else:
        model = DummyModel()

    # make agent
    agent = Agent(model, Configuration(env.configuration), make_setting(args))
    agent.postproc = check_timeout

    def run_agent(obs: Observation, _: Configuration):
        return agent(Observation(obs)).name

    return run_agent


def match(
    proc_number: int,
    timestamp: multiprocessing.managers.ValueProxy,
    model_path: pathlib.Path,
    args: argparse.Namespace
) -> Tuple[Dict[str, Any], List[List[List[int]]], List[int]]:
    random_seed(int(time.time()) + proc_number)

    LOGGER.info('start simulation [%d]: %s', proc_number, model_path)

    # make environment
    config = {'actTimeout': 20, 'runTimeout': 50000}
    env = make("hungry_geese", configuration=config, debug=args.debug)

    # make agent
    agent = make_agent(model_path, timestamp, env, args)

    try:
        env.run([agent, agent, agent, agent])

        if time.time() - timestamp.value > TIMEOUT:
            raise Exception('timeout')

        for s in env.state:
            if s['status'] != 'DONE':
                raise Exception('process is terminated by errors.')

        return (
            dict(env.configuration),
            convert_steps2records(env.steps),
            convert_state2result(env.state))
    except BaseException as e:
        LOGGER.critical(e)
        LOGGER.critical('\n%s', str(env.logs))
        LOGGER.critical(json.dumps(
            {'configuration': env.configuration, 'steps': env.steps}))
        raise e


def run_match(
    proc_number: int,
    receiver: multiprocessing.Queue,
    timestamp: multiprocessing.managers.ValueProxy,
    model_path: pathlib.Path,
    args: argparse.Namespace,
) -> None:
    try:
        receiver.put(match(proc_number, timestamp, model_path, args))
    except Exception as e:
        LOGGER.error(e)
        receiver.put([])


def wait_results(
    games: int,
    reveiver: queue.Queue,
    timestamp: multiprocessing.managers.ValueProxy,
) -> List[Tuple[Dict[str, Any], List[List[List[int]]], List[int]]]:
    records: List[Tuple[Dict[str, Any], List[List[List[int]]], List[int]]] = []

    while len(records) < games:
        timestamp.value = int(time.time())

        try:
            r: Tuple[Dict[str, Any], List[List[List[int]]], List[int]]
            r = reveiver.get(True, timeout=1.0)
            records.append(r)
        except queue.Empty:
            continue

    return records


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.debug)

    # files
    data_path = pathlib.Path(__file__).parent.parent / 'data'
    record_numbers = get_newest_record_numbers(data_path)
    model_numbers = get_newest_model_numbers(data_path)

    if len(record_numbers) != 0:
        record_path = data_path / f'{record_numbers[0]+1:05d}_record.pkl'
    else:
        record_path = data_path / '00000_record.pkl'

    if len(model_numbers) != 0:
        model_path = data_path / f'{model_numbers[0]:05d}_model.pkl'
    else:
        model_path = data_path / 'dummy_model.pkl'

    # run workers
    manager = multiprocessing.Manager()
    timestamp = manager.Value('i', int(time.time()))
    reveiver = manager.Queue()

    with multiprocessing.Pool(args.workers, setup_logging, (args.debug,)) as pool:
        params = [(i, reveiver, timestamp, model_path, args) for i in range(args.games)]
        pool.starmap_async(run_match, params)
        records = wait_results(args.games, reveiver, timestamp)

    # save records
    LOGGER.info('%s training data are created', len(records))
    LOGGER.info('save training data: %s', record_path)

    with open(record_path, 'wb') as writer:
        pickle.dump(records, writer)


if __name__ == '__main__':
    main()
