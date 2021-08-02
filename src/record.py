import argparse
import pickle
from typing import Any, Dict, List

from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration
from kaggle_environments.utils import Struct, structify

from agent import Setting, World
from model import DummyModel
from utils import setup_logging


def convert_steps2records(steps: List[Struct]) -> List[List[List[int]]]:
    '''Structure of a record.
    [
        segment of player 1,
        segment of player 2,
        segment of player 3,
        segment of player 4,
        foods,
        actions,
        gains,
    ]
    '''
    records: List[List[List[int]]] = []
    actions = ['NORTH', 'EAST', 'SOUTH', 'WEST']

    for prev_step, next_step in zip(steps[:-1], steps[1:]):
        record: List[List[int]] = [[], [], [], [], [], [-1] * 4, [0] * 4]

        for pidx in range(4):
            if len(prev_step[0]['observation']['geese'][pidx]) == 0:
                continue

            record[pidx].extend(prev_step[0]['observation']['geese'][pidx])

            if len(next_step[0]['observation']['geese'][pidx]) != 0:
                record[5][pidx] = actions.index(next_step[pidx]['action'])
                record[6][pidx] = int(
                    next_step[0]['observation']['geese'][pidx][0]
                    in prev_step[0]['observation']['food'])

        record[4].extend(prev_step[0]['observation']['food'])
        records.append(record)

    return records


def convert_records2steps(records: List[List[List[int]]]) -> List[Struct]:
    steps = []
    actions = ('NORTH', 'EAST', 'SOUTH', 'WEST')
    moves = ['NORTH', 'NORTH', 'NORTH', 'NORTH']

    for sidx, record in enumerate(records):
        step: List[Dict[str, Any]] = []

        for pidx in range(4):
            step.append({
                'action': moves[pidx],
                'reward': 0,
                'info': {},
                'observation': {'remainingOverageTime': 60, 'index': pidx},
                'status': 'ACTIVE' if len(record[pidx]) != 0 else 'DONE',
            })

        step[0]['observation']['step'] = sidx
        step[0]['observation']['food'] = list(record[4])
        step[0]['observation']['geese'] = []

        for pidx in range(4):
            step[0]['observation']['geese'].append(list(record[pidx]))

        for pidx in range(4):
            if record[5][pidx] != -1:
                moves[pidx] = actions[record[5][pidx]]

        steps.append(structify(step))

    return steps


def convert_state2result(state: Struct) -> List[int]:
    places = [[-1, i, s['reward']] for i, s in enumerate(state)]
    places.sort(key=lambda x: -x[2])
    places = [[p, i, r] for p, (_, i, r) in enumerate(places)]

    for i, j in zip(range(3), range(1, 4)):
        if places[i][2] == places[j][2]:
            places[j][0] = places[i][0]

    places.sort(key=lambda x: x[1])

    return [p for p, _, _ in places]


parser = argparse.ArgumentParser(
    description='View a record',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('record', type=str, help='Record file name.')
parser.add_argument('index', type=int, help='Index number of record.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.debug)

    with open(args.record, 'rb') as reader:
        config, records, result = pickle.load(reader)[args.index]

    steps = convert_records2steps(records)
    world = World(DummyModel(), Configuration(config), Setting())
    actions = ('NORTH', 'EAST', 'SOUTH', 'WEST', 'NONE')

    for i, (record, step) in enumerate(zip(records, steps)):
        world.apply(step[0]['observation'], use_model=False, update_move=False)
        print(f'step: {i}')
        print(world)
        print('action: {}'.format(', '.join(actions[v] for v in record[5])))
        print('values: {}'.format(', '.join(str(v) for v in record[6])))

    print('result: {}'.format(', '.join(f'{v}' for v in result)))


if __name__ == '__main__':
    main()
