'''This script evaluates models through games.
Each game is performed with 4 different models which are the newest or the strongest.
The most win model is saved as the strongest model (data/99999_model.pkl).
'''
import argparse
import logging
import multiprocessing.managers
import pathlib
import queue
import time
from typing import Callable, List, Tuple
import shutil

import numpy as np
import torch
import torch.nn as nn
from kaggle_environments import make
from kaggle_environments.core import Environment
from kaggle_environments.envs.hungry_geese.hungry_geese \
    import Action, Configuration, Observation

from agent import Agent
from utils import add_setting_arguments, get_newest_model_numbers, make_setting, random_seed, setup_logging

parser = argparse.ArgumentParser(
    description='Match models.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--iteration', type=int, default=400, help='Number of iterations.')
parser.add_argument('--threshold', type=int, default=0, help='Number of required wins.')
parser.add_argument('--workers', type=int, default=1, help='Number of workers.')
add_setting_arguments(parser)
parser.set_defaults(timelimit='100')
parser.set_defaults(search='ucbm')
parser.set_defaults(lookahead=3)
parser.add_argument('--seed', type=int, default=None, help='Random seed.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')


LOGGER = logging.getLogger(__name__)

# workers are terminated TIMEOUT seconds after the main process stops.
TIMEOUT = 5.0


def make_agent(
    model: nn.Module,
    timestamp: multiprocessing.managers.ValueProxy,
    env: Environment,
    args: argparse.Namespace,
) -> Callable:
    '''Create an agent.
    Args:
        model_path: file path of a cnn model.
        timestamp: timestamp object which will update by the main process.
        env: game environment object.
        args: command parameters.
    Returns:
        created agent.
    '''
    # make agent
    agent = Agent(model, Configuration(env.configuration), make_setting(args))

    # set timeout checker
    def check_timeout(agent: Agent, obs: Observation, action: Action) -> None:
        if time.time() - timestamp.value > TIMEOUT:
            raise Exception('timeout')

    agent.postproc = check_timeout

    # return agent
    def run_agent(obs: Observation, _: Configuration):
        return agent(Observation(obs)).name

    return run_agent


def match(
    proc_number: int,
    timestamp: multiprocessing.managers.ValueProxy,
    models: List[str],
    args: argparse.Namespace,
) -> List[int]:
    '''Perform a game and return the results.
    Args:
        proc_number: process number.
        timestamp: timestamp object which will update by the main process.
        model_path: file path of a cnn model.
        args: command parameters.
    Returns:
        list of rewards
    '''
    # set random seed for this match
    random_seed(args.seed + proc_number)

    # make environment
    env = make(
        "hungry_geese",
        configuration={'actTimeout': 20, 'runTimeout': 50000},
        debug=args.debug)

    # make agents
    agents = [make_agent(torch.jit.load(m), timestamp, env, args) for m in models]

    # run
    LOGGER.debug('run game: %d', proc_number)
    env.run(agents)
    LOGGER.debug('end game: %s', env.state)

    if time.time() - timestamp.value > TIMEOUT:
        raise Exception('timeout')

    # check
    for s in env.state:
        if s['status'] != 'DONE':
            LOGGER.error(env.logs)
            return []

    # send result
    return [s['reward'] for s in env.state]


def run_match(
    proc_number: int,
    receiver: multiprocessing.Queue,
    timestamp: multiprocessing.managers.ValueProxy,
    models: List[str],
    args: argparse.Namespace,
) -> None:
    '''Perform a game and send the results via the specified queue.
    Args:
        proc_number: process number.
        receiver: queue for sending the game records and the game results.
        timestamp: timestamp object which will update by the main process.
        model_path: file path of a cnn model.
        args: command parameters.
    '''
    try:
        receiver.put(match(proc_number, timestamp, models, args))
    except Exception as e:
        LOGGER.error(e)
        receiver.put([])


def wait_results(
    iteration: int,
    reveiver: queue.Queue,
    timestamp: multiprocessing.managers.ValueProxy,
) -> Tuple[np.ndarray, np.ndarray]:
    '''Wait for receiving game results.
    The game results are received from workers via queue.
    This returns the collection of the received results.
    Args:
        games: number of games.
        receiver: queue for receiving game results.
        timestamp: timestamp for notifying main process available.
    Returns:
        game results, game rewards.
    '''
    results = np.zeros([4, 4], dtype=np.int32)
    rewards = np.zeros([4], dtype=np.int32)
    points = np.array([3, 2, 1, 0], dtype=np.int32)

    while results[0].sum() < iteration:
        timestamp.value = int(time.time())

        try:
            rs: List[int] = reveiver.get(True, timeout=1.0)
        except queue.Empty:
            continue

        if len(rs) == 0:
            break

        players = np.array(rs).argsort()[::-1]
        places = list(range(4))

        for i, j in zip(range(3), range(1, 4)):
            if rs[players[i]] == rs[players[j]]:
                places[j] = places[i]

        results[players, places] += 1
        rewards += rs
        LOGGER.info('score[iter=%d] %s', results[0].sum(), results.dot(points))

    return results, rewards


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.debug)

    if args.seed is None:
        args.seed = int(time.time())

    # files
    data_path = pathlib.Path(__file__).parent.parent / 'data'

    with open(data_path / 'best_model.txt', 'r') as reader:
        best_numbers = [int(n) for n in reader.readline().split(',')]
        newest_number = int(reader.readline())

    model_numbers = best_numbers[::-1]

    for number in get_newest_model_numbers(data_path, size=4, max_number=99998):
        if number > newest_number:
            model_numbers.append(number)

    newest_number = max(model_numbers)
    model_numbers = model_numbers[-4:]

    # load models
    models = []

    for num in model_numbers:
        path = data_path / f'{num:05d}_model.pkl'
        LOGGER.info('load: %s', path)
        models.append(str(path))

    # run matches
    manager = multiprocessing.Manager()
    timestamp = manager.Value('i', int(time.time()))
    reveiver = manager.Queue()

    with multiprocessing.Pool(args.workers, setup_logging, (args.debug,)) as pool:
        params = [(i, reveiver, timestamp, models, args) for i in range(args.iteration)]
        pool.starmap_async(run_match, params)
        results, rewards = wait_results(args.iteration, reveiver, timestamp)

    # scores
    points = np.array([[3, 2, 1, 0]], dtype=np.int32)
    scores = (results * points).sum(axis=1)
    places = [(m, s, results[i]) for i, (m, s) in enumerate(zip(model_numbers, scores))]
    places.sort(key=lambda x: -x[1])

    if places[0][2][0] >= args.threshold:
        best_numbers = [n for n, _, _ in places]

    # print results
    with open(data_path / 'best_model.txt', 'w') as writer:
        writer.write('{}\n'.format(','.join(str(n) for n in best_numbers)))
        writer.write(f'{newest_number}\n')

        for pidx in range(4):
            writer.write('model[{:05d}] place=[{}], score={}, reward={}\n'.format(
                model_numbers[pidx], ','.join(f'{v:3d}' for v in results[pidx]),
                scores[pidx], rewards[pidx]))

    # copy the best model
    model_path = data_path / f'{best_numbers[0]:05d}_model.pkl'
    LOGGER.info('best model: %s', model_path)
    shutil.copy(model_path, data_path / '99999_model.pkl')


if __name__ == '__main__':
    main()
