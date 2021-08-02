import itertools
import logging
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from kaggle_environments.envs.hungry_geese.hungry_geese \
    import Action, Configuration, Observation
from numba import njit

from config import INPUT_AREA_SIZE, INPUT_CHANNELS, MAX_SEARCH
from config import EAST, INVALID, NORTH, SOUTH, WEST

'''
field data:
[00] player1 head
[01] player1 tail1
[02] player1 tail2
[03] player1 tail3
[04] player1 tail4
[05] player1 tail5
[06] player1 tail6
[07] player1 tail7
[08] player1 tail8..
[09] player1 nexts
[10] player1 bodies
[11] certain collision
[12] potential collision
[13-51] repeat for each player
[52] food

input data:
[00] my head
[01] my tail 1
[02] my tail 2
[03] my tail 3
[04] my tail 4
[05] my tail 5
[06] my tail 6
[07] my tail 7
[08] my tail 8..
[09] opponent1 head
[10] opponent1 tail 1
[11] opponent1 tail 2
[12] opponent1 tail 3
[13] opponent1 tail 4
[14] opponent1 tail 5
[15] opponent1 tail 6
[16] opponent1 tail 7
[17] opponent1 tails 8..
[18] opponent1 nexts
[19] opponent2 head
[20] opponent2 tail 1
[21] opponent2 tail 2
[22] opponent2 tail 3
[23] opponent2 tail 4
[24] opponent2 tail 5
[25] opponent2 tail 6
[26] opponent2 tail 7
[27] opponent2 tails 8..
[28] opponent2 nexts
[29] opponent3 head
[30] opponent3 tail 1
[31] opponent3 tail 2
[32] opponent3 tail 3
[33] opponent3 tail 4
[34] opponent3 tail 5
[35] opponent3 tail 6
[36] opponent3 tail 7
[37] opponent3 tails 8..
[38] opponent3 nexts
[39] any opponent head
[40] any opponent tail 1
[41] any opponent tail 2
[42] any opponent tail 3
[43] any opponent tail 4
[44] any opponent tail 5
[45] any opponent tail 6
[46] any opponent tail 7
[47] any opponent tails 8..
[48] any opponent nexts
[49] certain collisions (include myself)
[50] potential collisions
[51] length difference (-7..)  (set flag at the head)
[52] length difference (-6)
[53] length difference (-5)
[54] length difference (-4)
[55] length difference (-3)
[56] length difference (-2)
[57] length difference (-1)
[58] length difference (0)
[59] length difference (+1)
[60] length difference (+2)
[61] length difference (+3)
[62] length difference (+4)
[63] length difference (+5)
[64] length difference (+6)
[65] length difference (+7..)
[66] food
[67] remaining steps to hungry [1]
[68] remaining steps to hungry [2]
[69] remaining steps to hungry [3]
[70] remaining steps to hungry [4]
[71] remaining steps to hungry [5]
[72] remaining steps to hungry [6]
[73] remaining steps to hungry [7]
[74] remaining steps to hungry [8..]
[75] remaining steps to the end [1]
[76] remaining steps to the end [2]
[77] remaining steps to the end [3]
[78] remaining steps to the end [4]
[79] remaining steps to the end [5]
[80] remaining steps to the end [6]
[81] remaining steps to the end [7]
[82] remaining steps to the end [8..]
[83] step (000-019)
[84] step (020-039)
[85] step (040-059)
[86] step (060-079)
[87] step (080-099)
[88] step (100-119)
[89] step (120-139)
[90] step (140-159)
[91] step (160-179)
[92] step (180-199)
[93] direction (vertical)
[94] direction (horizontal)
[95] remaining players [2]
[96] remaining players [3]
[97] remaining players [4]

ouptut data:
[00] move left
[01] move straight
[02] move right
[03] value of 1st place
[04] value of 2nd place
[05] value of 3rd place

moves: [NORTH, EAST, SOUTH, WEST]
[00] value 1st
[01] value 2nd
[02] value 3rd
[03] value^2 1st (for ucb1-tuned)
[04] value^2 2nd (for ucb1-tuned)
[05] value^2 3rd (for ucb1-tuned)
[06] visit
[07] alive
[08] enabled
[09] policy (for puct or zero)
[10] expanded (1: not expanded, 0: expanded)

extra:
[00] minimum extra growth (player 1)
[01] minimum extra growth (player 2)
[02] minimum extra growth (player 3)
[03] minimum extra growth (player 4)
[04] maximum extra growth
'''

LOGGER = logging.getLogger(__name__)

# _BEGIN_AGENT_
FIELD_FOOD = 52
FS = 13  # field size
FN = 9  # field offset of next heads
FB = 10  # field offset of bodies
FC = 11  # field offset of certain collision
FP = 12  # field offset of potential collisioin

INPUT_COLLISION = 49
INPUT_DIFFERENCE = 58
INPUT_FOOD = 66
INPUT_HUNGRY = 67
INPUT_END = 75
INPUT_STEP = 83
INPUT_DIRECTION = 93
INPUT_PLAYERS = 95

MOVE_VISIT = 6
MOVE_ALIVE = 7
MOVE_ENABLED = 8
MOVE_POLICY = 9
MOVE_EXPANDED = 10

# stack(for lookahead): [step, y, x, direction, move, backup]:
AS_Y = 4
AS_X = 5
AS_NMOVE = 6
AS_PMOVE = 7
AS_BACK = 8


@njit
def get_direction(prev: int, current: int, rows: int, columns: int) -> int:
    prev_y, prev_x = row_col(prev, columns)
    curr_y, curr_x = row_col(current, columns)
    move_x = (curr_x - prev_x) % columns
    move_y = (curr_y - prev_y) % rows

    if move_y == rows - 1:
        return NORTH
    elif move_x == 1:
        return EAST
    elif move_y == 1:
        return SOUTH
    elif move_x == columns - 1:
        return WEST
    else:
        return INVALID


@njit
def row_col(position: int, columns: int) -> Tuple[int, int]:
    return position // columns, position % columns


@njit
def get_next_posision(position: int, direction: int, rows: int, columns: int) -> int:
    y, x = row_col(position, columns)

    if direction == NORTH:
        y -= 1
        y %= rows
    elif direction == EAST:
        x += 1
        x %= columns
    elif direction == SOUTH:
        y += 1
        y %= rows
    elif direction == WEST:
        x -= 1
        x %= columns

    return y * columns + x


def get_move_values(moves: np.ndarray, safe: float, player: int) -> np.ndarray:
    return (moves[:, :, 0] * (1.0 - safe)) + (moves[:, :, max(player - 2, 0)] * safe)


def get_move_values2(moves: np.ndarray, safe: float, player: int) -> np.ndarray:
    return (moves[:, :, 3] * (1.0 - safe)) + (moves[:, :, max(player + 1, 3)] * safe)


def select_action_by_policy(moves: np.ndarray) -> np.ndarray:
    return np.argmax((moves[:, :, MOVE_POLICY]) * moves[:, :, MOVE_ENABLED], axis=1)


def select_actions_by_beta(moves: np.ndarray, safe: float, player: int) -> np.ndarray:
    values = get_move_values(moves, safe, player)
    alpha = np.minimum(values, moves[:, :, MOVE_VISIT])
    beta = moves[:, :, MOVE_VISIT] - alpha
    values = np.random.beta(alpha + 1e-6, beta + 1e-6)

    return np.argmax(values * moves[:, :, MOVE_ENABLED], axis=1)


def select_actions_by_ucbm(moves: np.ndarray, safe: float, player: int) -> np.ndarray:
    '''Modified UCB (not UCB1)'''
    values = get_move_values(moves, safe, player) / (moves[:, :, MOVE_VISIT] + 1e-6)
    visits = np.sqrt(1.0 / (moves[:, :, MOVE_VISIT] + 1e-6))

    return np.argmax(values * visits * moves[:, :, MOVE_ENABLED], axis=1)


def select_actions_by_ucbr(moves: np.ndarray, safe: float, player: int) -> np.ndarray:
    '''Randamized-Modified UCB (not UCB1)'''
    values = get_move_values(moves, safe, player) / (moves[:, :, MOVE_VISIT] + 1e-6)
    visits = np.sqrt((1.0 + np.random.rand(4, 1)) / (moves[:, :, MOVE_VISIT] + 1e-6))

    return np.argmax(values * visits * moves[:, :, MOVE_ENABLED], axis=1)


def select_actions_by_ucb1(moves: np.ndarray, safe: float, player: int) -> np.ndarray:
    values = get_move_values(moves, safe, player) / (moves[:, :, MOVE_VISIT] + 1e-6)
    totals = (moves[:, :, MOVE_VISIT].sum(axis=1) + 1)[:, None]
    visits = np.sqrt(2 * np.log(totals) / (moves[:, :, MOVE_VISIT] + 1e-6))

    return np.argmax((values + visits) * moves[:, :, MOVE_ENABLED], axis=1)


def select_actions_by_puct(moves: np.ndarray, safe: float, player: int) -> np.ndarray:
    '''PUCT(modified)'''
    values = get_move_values(moves, safe, player) / (moves[:, :, MOVE_VISIT] + 1e-6)
    totals = np.log(moves[:, :, MOVE_VISIT].sum(axis=1) + 1)[:, None]
    visits = 1.0 * np.sqrt(totals / (moves[:, :, MOVE_VISIT] + 1e-6))
    priority = values + moves[:, :, MOVE_POLICY] * visits

    return np.argmax(priority * moves[:, :, MOVE_ENABLED], axis=1)


def select_actions_by_zero(moves: np.ndarray, safe: float, player: int) -> np.ndarray:
    '''AlphaZero'''
    values = get_move_values(moves, safe, player) / (moves[:, :, MOVE_VISIT] + 1e-6)
    totals = moves[:, :, MOVE_VISIT].sum(axis=1, keepdims=True)
    visits = 5.0 * np.sqrt(totals) / (moves[:, :, MOVE_VISIT] + 1)
    priority = values + moves[:, :, MOVE_POLICY] * visits

    return np.argmax(priority * moves[:, :, MOVE_ENABLED], axis=1)


class Setting(object):
    def __init__(self, **kwargs) -> None:
        self.timelimit = 'auto'  # time limit of search moves ('auto', 'zero', 'none' or visit count)
        self.depthlimit = 20  # depth limit of search moves (simulation only at more than the specified depth)
        self.decision = 'hybrid'  # value for move decisions ('hybrid', 'value', 'visit')
        self.normalize = True  # normalize values
        self.collision = 0.0  # penalty of potetial collision moves (ignore potential collisions at 0.0)
        self.eating = 0.0  # value of eating a food (ignore foods at 0.0)
        self.search = 'ucb1'  # algrithm of next move selections ('beta' or 'ucb1')
        self.safe = 0.0  # safe parameter at move decisions.
        self.strict = False  # evaluate a value of the root node with other edge and leaf nodes.
        self.lookahead = 4  # depth of look-ahead by depth-first search (not inference by NN or MCTS).
        self.policybase = 0.1  # base of policy
        self.valuebase = 0.0  # base of value
        self.valuetail = 0.0  # base at neighbor of own tail
        self.usepolicy = True  # use policy model
        self.usevalue = True  # use value model
        self.random = 0.0  # probability of random decision (for reinforced learning)
        self.gpu = -1  # use gpu (use cpu at -1)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __and__(self, s: 'Setting') -> 'Setting':
        return Setting(**{
            k: v if v == getattr(s, k) else None for k, v in self.__dict__.items()})

    def __sub__(self, s: 'Setting') -> 'Setting':
        return Setting(**{
            k: None if v == getattr(s, k) else v for k, v in self.__dict__.items()})

    def __str__(self) -> str:
        return ', '.join(f'{k}={v}' for k, v in self.__dict__.items() if v is not None)


class World(object):
    def __init__(self, model: nn.Module, config: Configuration, setting: Setting) -> None:
        self.model = model
        self.config = config
        self.setting = setting
        # node depth
        self.depth = 0
        # visit count
        self.visit = 0
        # number of steps
        self.step = -1
        # game status
        self.finished = False
        # values after the end of the game
        self.rewards = np.zeros([4], dtype=np.int16)
        # previous head positions
        self.prevs = np.zeros([4], dtype=np.int16)
        # head directions
        self.directions = np.zeros([4], dtype=np.int16)
        # geese length
        self.lengths = np.zeros([4], dtype=np.int16)
        # geese segments
        self.segments = np.zeros([4, config.rows * config.columns], dtype=np.int16)
        # food positions
        self.foods = np.zeros([2], dtype=np.int16)
        # field map
        self.field = np.zeros([53, config.rows, config.columns], dtype=np.float32)
        # collision map
        self.collision = np.zeros([4, config.rows, config.columns], dtype=np.int8)
        # remaining lives
        self.lives = np.zeros([4, 4], dtype=np.int16)
        # feature array for inference
        self.input = np.zeros(
            [4, INPUT_CHANNELS, INPUT_AREA_SIZE, INPUT_AREA_SIZE], dtype=np.float32)
        # values of this step
        self.values = np.zeros([4, 3], dtype=np.float32)
        # values of this step
        self.policies = np.zeros([4, 3], dtype=np.float32)
        # values of moves
        self.moves = np.zeros([4, 4, 11], dtype=np.float32)
        # values of eating foods
        self.eatings = np.zeros([4], dtype=np.float32)
        # next nodes
        self.nexts: Dict[int, 'World'] = {}

        # setup model status
        if self.setting.gpu >= 0:
            self.model = self.model.cuda(device=self.setting.gpu)

        self.model.eval()

    def select_crush_action(self, pidx: int) -> int:
        '''Select the last action.
        The goose will be die at the next step, but avoids body crush.
        '''
        for action in (NORTH, EAST, SOUTH, WEST):
            if (action + 2) % 4 == self.directions[pidx]:
                continue

            pos = get_next_posision(
                self.segments[pidx, 0], action, self.config.rows, self.config.columns)

            if pos not in self.segments[pidx, :self.lengths[pidx]]:
                return action

        return 0

    def is_available_move(self, player_index: int, move: int) -> bool:
        return self._search_longest_path(player_index, move, 1) != 0

    def apply(
        self,
        obs: Observation,
        use_model: bool,
        update_move: bool,
        prevobs: Optional[Observation] = None
    ) -> None:
        '''Update world by an observation object as a root node.'''
        if prevobs is not None:
            self._apply(prevobs, use_model=False, update_move=False)

        self._apply(obs, use_model=use_model, update_move=update_move)

    def _apply(self, obs: Observation, use_model: bool, update_move: bool) -> None:
        # initialize for this search tree
        self.depth = 0
        self.visit = 0
        self.step = obs.step
        self.finished = False
        self.rewards.fill(0)
        self.eatings.fill(0)

        # save previous positions
        self.prevs[:] = self.segments[:, 0]

        # update segments
        for i, geese in enumerate(obs.geese):
            self.lengths[i] = len(geese)

            if len(geese) == 0:
                continue

            for j, pos in enumerate(geese):
                self.segments[i, j] = pos

        # update directions
        self.directions[:] = [
            get_direction(p, c, self.config.rows, self.config.columns)
            for p, c in zip(self.prevs, self.segments[:, 0])]

        # update foods
        self.foods.fill(-1)

        for fidx, food in enumerate(obs.food):
            self.foods[fidx] = food

        self._update_all(use_model=use_model, update_move=update_move)

    def expand(self, leaf: 'World') -> np.ndarray:
        '''Expand this tree and append a leaf node.
        '''
        self.visit += 1

        # select actions
        # leaf node is not created if it is too deep or the game is over.
        player = len([v for v in self.lengths if v != 0])

        if self.depth > self.setting.depthlimit or self.finished:
            return np.array(self.values)
        elif self.setting.search == 'policy':
            actions = select_action_by_policy(self.moves)
        elif self.setting.search == 'beta':
            actions = select_actions_by_beta(self.moves, self.setting.safe, player)
        elif self.setting.search == 'ucbm':
            actions = select_actions_by_ucbm(self.moves, self.setting.safe, player)
        elif self.setting.search == 'ucbr':
            actions = select_actions_by_ucbr(self.moves, self.setting.safe, player)
        elif self.setting.search == 'ucb1':
            actions = select_actions_by_ucb1(self.moves, self.setting.safe, player)
        elif self.setting.search == 'puct':
            actions = select_actions_by_puct(self.moves, self.setting.safe, player)
        elif self.setting.search == 'zero':
            actions = select_actions_by_zero(self.moves, self.setting.safe, player)
        else:
            raise Exception(f'unsupported algorithm: {self.setting.search}')

        # check lives and select move just before crush
        for pidx in range(4):
            if self.lengths[pidx] == 0:
                actions[pidx] = 0
            elif (self.moves[pidx, :, MOVE_ENABLED] == 0).all():
                actions[pidx] = self.select_crush_action(pidx)

        # check the tree
        action_key = sum(a << i * 2 for i, a in enumerate(actions))

        if action_key in self.nexts:
            values = self.nexts[action_key].expand(leaf)
        else:
            self.nexts[action_key] = leaf
            self._expand(leaf, actions)
            values = np.array(leaf.values)

        # apply penalties
        penalty = np.array([self.moves[p, a, MOVE_ALIVE] for p, a in enumerate(actions)])
        values *= penalty[:, None]

        # add node values
        for pidx, action in enumerate(actions):
            if self.setting.strict and self.moves[pidx, action, MOVE_EXPANDED] == 1.0:
                self.moves[pidx, action, 0:3] = values[pidx]
                self.moves[pidx, action, 3:6] = values[pidx] ** 2
                self.moves[pidx, action, MOVE_VISIT] = 1.0
                self.moves[pidx, action, MOVE_EXPANDED] = 0.0
            else:
                self.moves[pidx, action, 0:3] += values[pidx]
                self.moves[pidx, action, 3:6] += values[pidx] ** 2
                self.moves[pidx, action, MOVE_VISIT] += 1.0

        return values

    def _expand(self, leaf: 'World', actions: np.ndarray) -> None:
        # make a leaf node
        leaf.depth = self.depth + 1
        leaf.visit = 0
        leaf.step = self.step + 1
        leaf.finished = self.finished
        leaf.rewards[:] = self.rewards
        leaf.eatings[:] = self.eatings

        # set previous positions
        leaf.prevs[:] = self.segments[:, 0]

        # set next segments
        max_length = self.lengths.max()
        leaf.lengths[:] = self.lengths
        leaf.segments[:, 1:max_length + 1] = self.segments[:, :max_length]

        for pidx in range(4):
            if leaf.lengths[pidx] == 0:
                leaf.segments[pidx, 0] = leaf.segments[pidx, 1]
                continue

            leaf.segments[pidx, 0] = get_next_posision(
                leaf.segments[pidx, 1], actions[pidx], leaf.config.rows, leaf.config.columns)

        # set next directions
        leaf.directions[:] = [
            get_direction(p, c, leaf.config.rows, self.config.columns)
            for p, c in zip(leaf.prevs, leaf.segments[:, 0])]

        # check foods
        leaf.foods[:] = self.foods

        for fidx, pidx in itertools.product(range(2), range(4)):
            if leaf.lengths[pidx] == 0:
                continue
            elif leaf.segments[pidx, 0] == leaf.foods[fidx]:
                leaf.lengths[pidx] += 1
                leaf.eatings[pidx] += self.setting.eating
                leaf.foods[fidx] = -1

        # check collisions (body hit)
        collision = np.zeros([4, self.config.rows * self.config.columns], dtype=np.int8)

        for pidx in range(4):
            for i in range(leaf.lengths[pidx]):
                collision[pidx, leaf.segments[pidx, i]] += 1

        for pidx in range(4):
            if leaf.lengths[pidx] == 0:
                continue
            elif collision[pidx, leaf.segments[pidx, 0]] > 1:
                leaf.lengths[pidx] = 0
                collision[pidx] = 0

        # hungry turn (hungry reduction is preformed after collision check)
        if leaf.step % 40 == 0:
            for pidx in range(4):
                if leaf.lengths[pidx] == 0:
                    continue
                leaf.lengths[pidx] -= 1
                collision[pidx, leaf.segments[pidx, leaf.lengths[pidx]]] = 0

        # check collisions (goose collision)
        collision = collision.sum(axis=0)

        for pidx in range(4):
            if leaf.lengths[pidx] == 0:
                continue
            elif collision[leaf.segments[pidx, 0]] > 1:
                leaf.lengths[pidx] = 0

        # check the end of the game
        alive_players = len([v for v in leaf.lengths if v > 0])

        if alive_players <= 1 or leaf.step >= self.config.episode_steps - 1:
            leaf.finished = True

        # update the next world
        leaf._update_all(use_model=True, update_move=True)

    def _update_all(self, use_model: bool, update_move: bool) -> None:
        self.nexts.clear()
        self._update_rewards()
        self._update_collision()
        self._update_field()
        self._update_input()
        self._update_values(use_model)

        if not self.finished and update_move:
            self._update_lives()
            self._update_moves()
        else:
            self.lives[:] = 0.0
            self.moves[:] = 0.0

    def _update_rewards(self) -> None:
        for pidx in range(4):
            if self.lengths[pidx] == 0:
                continue

            self.rewards[pidx] = self.step * self.config.max_length
            self.rewards[pidx] += self.lengths[pidx]

    def _update_collision(self) -> None:
        # make collision map
        self.collision.fill(0)

        for pidx in range(4):
            for sidx in range(self.lengths[pidx]):
                y, x = row_col(self.segments[pidx, sidx], self.config.columns)
                self.collision[pidx, y, x] = self.lengths[pidx] - sidx

    def _update_field(self) -> None:
        def setone(array, index):
            y, x = row_col(index, self.config.columns)
            array[y, x] = 1.0

        # initialize
        self.field.fill(0.0)

        # check if potential head collisions are required
        # when only 2 geeses are alive, collisions of shorter one are not considered.
        head_collisions = [True, True, True, True]
        player_lengths = [
            (pidx, length) for pidx, length in enumerate(self.lengths)
            if length != 0]

        if len(player_lengths) == 2:
            if player_lengths[0][1] < player_lengths[1][1]:
                head_collisions[player_lengths[0][0]] = False
            elif player_lengths[0][1] > player_lengths[1][1]:
                head_collisions[player_lengths[1][0]] = False
            else:
                head_collisions[player_lengths[0][0]] = False
                head_collisions[player_lengths[1][0]] = False

        # foods
        for food in self.foods:
            if food != -1:
                setone(self.field[FIELD_FOOD], food)

        # geeses
        for pidx in range(4):
            if self.lengths[pidx] == 0:
                continue

            # head
            setone(self.field[pidx * FS + 0], self.segments[pidx, 0])

            # tails
            for sidx in range(self.lengths[pidx]):
                tail_index = min(self.lengths[pidx] - sidx, 8)
                setone(self.field[pidx * FS + tail_index], self.segments[pidx, sidx])

            # bodies
            self.field[pidx * FS + FB] = self.collision[pidx] > 1

            # check alive
            next_length = 0

            for direction in (NORTH, EAST, SOUTH, WEST):
                pos = get_next_posision(
                    self.segments[pidx, 0], direction,
                    self.config.rows, self.config.columns)
                y, x = row_col(pos, self.config.columns)

                if self.field[pidx * FS + FB, y, x] == 0.0:
                    next_length = self.lengths[pidx]
                    break

            if next_length <= 0:
                continue

            # certain collisions
            if self.step % 40 == 39:
                self.field[pidx * FS + FC] = self.collision[pidx] > 2
                next_length -= 1
            else:
                self.field[pidx * FS + FC] = self.collision[pidx] > 1

            # next heads, potential collision and move penalty
            for direction in (NORTH, EAST, SOUTH, WEST):
                pos = get_next_posision(
                    self.segments[pidx, 0], direction,
                    self.config.rows, self.config.columns)

                if pos == self.prevs[pidx]:
                    continue

                # next heads
                setone(self.field[pidx * FS + FN], pos)

                # potential collisions and move penalties
                if head_collisions[pidx]:
                    setone(self.field[pidx * FS + FP], pos)

                if pos in self.foods:
                    setone(self.field[pidx * FS + FP], self.segments[pidx, next_length - 1])

        # clear next heads and potential collisions where collisions occur certainly.
        collisions = [self.field[pidx * FS + FC] for pidx in range(4)]
        collisions = (np.stack(collisions, axis=0).sum(axis=0) == 0.0).astype(np.float32)

        for pidx in range(4):
            self.field[pidx * FS + FN] *= collisions
            self.field[pidx * FS + FP] *= collisions

    def _update_input(self) -> None:
        # clear input data
        self.input.fill(0.0)

        # update input array
        World_update_input_jit(
            self.input,
            self.lengths,
            self.segments,
            self.field,
            self.config.rows,
            self.config.columns,
            self.step,
            self.config.episode_steps,
        )

        # rotate input field
        for pidx in range(4):
            if self.lengths[pidx] == 0:
                continue
            elif self.directions[pidx] == EAST:
                self.input[pidx] = np.rot90(self.input[pidx], k=1, axes=(1, 2))
                self.input[pidx, INPUT_DIRECTION + 1] = 1.0
            elif self.directions[pidx] == SOUTH:
                self.input[pidx] = np.rot90(self.input[pidx], k=2, axes=(1, 2))
                self.input[pidx, INPUT_DIRECTION + 0] = 1.0
            elif self.directions[pidx] == WEST:
                self.input[pidx] = np.rot90(self.input[pidx], k=3, axes=(1, 2))
                self.input[pidx, INPUT_DIRECTION + 1] = 1.0
            else:
                self.input[pidx, INPUT_DIRECTION + 0] = 1.0

    def _update_values(self, use_model: bool) -> None:
        # initialize
        self.moves.fill(0.0)

        # prediction
        if not use_model or self.finished:
            # default value
            self.values[:] = 1.0
            self.policies[:] = 1.0
        else:
            # inference
            players = [pidx for pidx, length in enumerate(self.lengths) if length != 0]
            inputs = np.stack([self.input[pidx] for pidx in players])
            inputs = torch.tensor(inputs)

            if self.setting.gpu >= 0:
                inputs = inputs.cuda(device=self.setting.gpu)

            with torch.no_grad():
                outputs = self.model(inputs).cpu()

            # make prediction array
            predictions = np.zeros([4, 6], dtype=np.float32)

            for i, pidx in enumerate(players):
                predictions[pidx] = outputs[i]

            # update policies and values
            self.policies[:] = predictions[:, 0:3]
            self.values[:] = predictions[:, 3:6]

            # reset unsused values
            if not self.setting.usepolicy:
                self.policies[:] = 1.0

            if not self.setting.usevalue:
                self.values[:] = 1.0

            # apply policy base
            policybase = self.setting.policybase
            self.policies[:] = self.policies * (1.0 - policybase) + policybase

            # apply eatings
            self.values[:] = np.clip(self.values + self.eatings[:, None], 1e-6, 1.0)

            # apply value base
            valuebase = np.zeros([4, 1], dtype=np.float32)

            for pidx in range(4):
                if self.lengths[pidx] != 0:
                    valuebase[pidx, 0] = self.setting.valuebase

            self.values[:] = self.values * (1.0 - valuebase) + valuebase

            # apply value of tail neighbor
            valuetails = np.zeros([4, 1], dtype=np.float32)

            for pidx in range(4):
                if self.lengths[pidx] == 0:
                    continue

                for action in (NORTH, EAST, SOUTH, WEST):
                    if (action + 2) % 4 == self.directions[pidx]:
                        continue

                    next_position = get_next_posision(
                        self.segments[pidx, 0], action,
                        self.config.rows, self.config.columns)

                    if next_position == self.segments[pidx, self.lengths[pidx] - 1]:
                        valuetails[pidx, 0] = self.setting.valuetail
                        break

            self.values[:] = self.values * (1.0 - valuetails) + valuetails

        # set fixed values
        if not self.finished:
            remains = len([v for v in self.lengths if v != 0])
        else:
            remains = 0

        if remains < 4:
            places = [[i, 0, r] for i, r in enumerate(self.rewards)]
            places.sort(key=lambda x: -x[2])

            for i in range(1, 4):
                if places[i][2] == places[i - 1][2]:
                    places[i][1] = places[i - 1][1]
                else:
                    places[i][1] = i

            for pidx, place, _ in places:
                if place < remains:
                    self.values[pidx, remains - 1:] = 1.0
                else:
                    self.values[pidx, :place] = 0.0
                    self.values[pidx, place:] = 1.0

        # notmalize
        if self.setting.normalize:
            self.values /= self.values.sum(axis=0, keepdims=True) + 1e-6

    def _update_lives(self) -> None:
        if self.step >= self.config.episode_steps - 1:
            self.lives[:] = 0
            return

        # check player lives
        self._update_live_map()

        # check move lives
        lives = self.lives.max(axis=1)
        self._update_live_map(lives)

    def _update_live_map(self, lives: Optional[np.ndarray] = None) -> None:
        remain_step = self.config.episode_steps - self.step - 1
        search_depth = min(remain_step, self.setting.lookahead)

        for pidx in range(4):
            if self.lengths[pidx] == 0:
                self.lives[pidx] = 0
                continue

            for move in range(4):
                if (move + 2) % 4 == self.directions[pidx]:
                    self.lives[pidx, move] = 0
                    continue

                distance = self._search_longest_path(pidx, move, search_depth, lives)

                if distance == search_depth:
                    self.lives[pidx, move] = remain_step
                else:
                    self.lives[pidx, move] = distance

    def _update_moves(self) -> None:
        # update moves
        for pidx in range(4):
            if self.lengths[pidx] == 0:
                self.moves[pidx, :, MOVE_ENABLED] = 0.0
                continue

            if self.directions[pidx] == EAST:
                midxs = [0, 1, 2, 3]
            elif self.directions[pidx] == SOUTH:
                midxs = [1, 2, 3, 0]
            elif self.directions[pidx] == WEST:
                midxs = [2, 3, 0, 1]
            else:  # NORTH
                midxs = [3, 0, 1, 2]

            for vidx, midx in enumerate(midxs[:3]):
                alive = 1.0 - self._get_penalty(pidx, midx)
                policy = self.policies[pidx, vidx]
                enabled = 1.0 if alive != 0.0 else 0.0

                if self.setting.search in ('puct', 'zero'):
                    initvalue = self.values[pidx] * alive
                else:
                    initvalue = self.values[pidx] * alive * policy

                self.moves[pidx, midx, :3] = initvalue
                self.moves[pidx, midx, MOVE_VISIT] = 1.0
                self.moves[pidx, midx, MOVE_ALIVE] = alive
                self.moves[pidx, midx, MOVE_ENABLED] = enabled
                self.moves[pidx, midx, MOVE_POLICY] = policy
                self.moves[pidx, midx, MOVE_EXPANDED] = 1.0

            self.moves[pidx, midxs[3], MOVE_ENABLED] = 0.0

        # set power 2 of value
        self.moves[:, :, 3:6] = self.moves[:, :, 0:3] ** 2

    def _get_penalty(self, player_index: int, move: int) -> float:
        # check path
        max_life = self.lives[player_index].max()
        if max_life == 0 or self.lives[player_index, move] != max_life:
            return 1.0

        # check collision
        pos = get_next_posision(
            self.segments[player_index, 0], move,
            self.config.rows, self.config.columns)
        y, x = row_col(pos, self.config.columns)
        penalty = 0.0

        for oidx in range(4):
            if oidx != player_index and self.field[oidx * FS + FP, y, x] != 0.0:
                collision = self.setting.collision if self.step != 0 else 0.9
                penalty += (1.0 - penalty) * collision

        return penalty

    def _search_longest_path(
        self,
        player_index: int,
        move: int,
        depth: int,
        lives: Optional[np.ndarray] = None
    ) -> int:
        if lives is None:
            lives = np.array([self.config.episode_steps] * 4, np.int16)

        # collision map
        return World_search_longest_path_jit(
            player_index, move, depth, lives,
            self.segments, self.lengths, self.field, self.collision,
            self.step, self.config.rows, self.config.columns)

    def __str__(self) -> str:
        def position2char(array: np.ndarray):
            if array[FIELD_FOOD] == 1.0:
                return ' F '

            for idx, value in enumerate(array):
                if value == 0.0:
                    continue
                elif idx % FS == 0:
                    return f'[{idx // FS}]'
                elif idx % FS == 1:
                    return f'<{idx // FS}>'
                elif idx % FS == 2:
                    return f'({idx // FS})'
                elif idx % FS == 3:
                    return f'|{idx // FS}|'
                elif idx % FS == 4:
                    return f';{idx // FS};'
                elif idx % FS == 5:
                    return f':{idx // FS}:'
                elif idx % FS == 6:
                    return f',{idx // FS},'
                elif idx % FS == 7:
                    return f'.{idx // FS}.'
                elif idx % FS == 8:
                    return f' {idx // FS} '

            for idx, value in enumerate(array):
                if value == 0.0:
                    continue
                elif idx % FS == FN:
                    return ' * '

            return ' . '

        def collision2char(array: np.ndarray):
            for idx, value in enumerate(array):
                if value == 0.0:
                    continue
                elif idx % FS == FC:
                    return ' # '
                elif idx % FS == FP:
                    return ' * '

            for idx, value in enumerate(array):
                if value == 0.0:
                    continue
                elif idx % FS == FB:
                    return ' @ '

            return ' . '

        texts = []

        # texts.append('positions:')
        for y in range(self.config.rows):
            texts.append(''.join(
                position2char(self.field[:, y, x]) for x in range(self.config.columns)))

        # texts.append('collisions:')
        # for y in range(self.config.rows):
            # texts.append(''.join(
            # collision2char(self.field[:, y, x]) for x in range(self.config.columns)))

        return '\n'.join(texts)


@njit
def World_update_input_jit(
    input_array: np.ndarray,
    lengths: np.ndarray,
    segments: np.ndarray,
    field: np.ndarray,
    rows: int,
    columns: int,
    step: int,
    episode_steps: int,
) -> None:
    # make places
    places = [(i, length) for i, length in enumerate(lengths)]
    places.sort(key=lambda x: -x[1])

    # make input data
    for pidx in range(4):
        if lengths[pidx] == 0:
            continue

        # each head is located at the center of input map
        # each edge is padded repeatedly
        head_y, head_x = row_col(segments[pidx, 0], columns)
        offset_x = (INPUT_AREA_SIZE // 2) - head_x
        offset_y = (INPUT_AREA_SIZE // 2) - head_y

        while offset_x > 0:
            offset_x -= columns

        while offset_y > 0:
            offset_y -= rows

        for y in range(offset_y, INPUT_AREA_SIZE, rows):
            for x in range(offset_x, INPUT_AREA_SIZE, columns):
                src_bx, src_by = max(-x, 0), max(-y, 0)
                dst_bx, dst_by = max(x, 0), max(y, 0)
                src_ex = min(src_bx + (INPUT_AREA_SIZE - dst_bx), columns)
                src_ey = min(src_by + (INPUT_AREA_SIZE - dst_by), rows)
                dst_ex = dst_bx + (src_ex - src_bx)
                dst_ey = dst_by + (src_ey - src_by)

                # apply features of the player
                input_array[pidx, 0:9, dst_by:dst_ey, dst_bx:dst_ex]\
                    = field[pidx * FS:pidx * FS + 9, src_by:src_ey, src_bx:src_ex]
                input_array[pidx, INPUT_COLLISION + 0, dst_by:dst_ey, dst_bx:dst_ex] \
                    += field[pidx * FS + FB, src_by:src_ey, src_bx:src_ex]

                # apply features of other players
                for place, oidx in enumerate([i for i, _ in places if i != pidx]):
                    if lengths[oidx] == 0:
                        continue

                    # features
                    offset = 9 + place * 10
                    input_array[pidx, offset:offset + 10, dst_by:dst_ey, dst_bx:dst_ex]\
                        = field[oidx * FS:oidx * FS + 10, src_by:src_ey, src_bx:src_ex]
                    input_array[pidx, INPUT_COLLISION + 0, dst_by:dst_ey, dst_bx:dst_ex] \
                        += field[oidx * FS + FC, src_by:src_ey, src_bx:src_ex]
                    input_array[pidx, INPUT_COLLISION + 1, dst_by:dst_ey, dst_bx:dst_ex] \
                        += field[oidx * FS + FP, src_by:src_ey, src_bx:src_ex]

                    # differences
                    diff_y, diff_x = row_col(segments[oidx, 0], columns)
                    diff_y, diff_x = diff_y + y, diff_x + x

                    if 0 <= diff_x < INPUT_AREA_SIZE and 0 <= diff_y < INPUT_AREA_SIZE:
                        diff = max(min(lengths[oidx] - lengths[pidx], 7), -7)
                        input_array[pidx, INPUT_DIFFERENCE + diff, diff_y, diff_x] = 1.0

                # foods
                input_array[pidx, INPUT_FOOD, dst_by:dst_ey, dst_bx:dst_ex]\
                    = field[FIELD_FOOD, src_by:src_ey, src_bx:src_ex]

        # aggregate opponent positions
        for oidx in range(3):
            offset = 9 + oidx * 10
            input_array[pidx, 39:49] += input_array[pidx, offset:offset + 10]

    # step values
    input_array[:, INPUT_HUNGRY + min(39 - step % 40, 7)] = 1.0
    input_array[:, INPUT_END + max(min(episode_steps - step - 2, 7), 0)] = 1.0
    input_array[:, INPUT_STEP + min(step // 20, 19)] = 1.0
    input_array[:, INPUT_PLAYERS + len([v for v in lengths if v != 0]) - 2] = 1.0

    # clip values
    input_array[:] = np.minimum(input_array, 1.0)


@njit
def World_search_longest_path_jit(
    player_index: int,
    move: int,
    depth: int,
    lives: np.ndarray,
    segments: np.ndarray,
    lengths: np.ndarray,
    field: np.ndarray,
    collision: np.ndarray,
    step: int,
    rows: int,
    columns: int,
) -> int:
    index = 0
    distance = 0

    # copy collision array
    collision = collision.copy()

    # next position
    pos = get_next_posision(segments[player_index, 0], move, rows, columns)
    y, x = row_col(pos, columns)

    # [step1, step2, step3, step4, y, x, direction, move, backup]
    stack = np.zeros((depth, 9), dtype=np.int8)
    stack[0] = [1, 1, 1, 1, y, x, -1, move, 0]

    if step % 40 == 39:  # hungry turn
        for oidx in [p for p in range(4) if p != player_index]:
            stack[0, oidx] += 1

    # search path
    while index >= 0:
        # check collision
        if stack[index, AS_NMOVE] == -1:
            y = stack[index, AS_Y]
            x = stack[index, AS_X]
            crush = False

            for pidx in range(4):
                if lives[pidx] >= index and collision[pidx, y, x] > stack[index, pidx]:
                    crush = True
                    break

            if crush:
                index -= 1
                continue

            if index + 1 == depth:
                return depth
            else:
                distance = max(distance, index + 1)

            stack[index, AS_NMOVE] = 0
            stack[index, AS_BACK] = collision[player_index, y, x]
            collision[player_index, y, x] = lengths[player_index] + index + 1

        # check move
        if stack[index, AS_NMOVE] == 4:
            collision[player_index, stack[index, AS_Y], stack[index, AS_X]] \
                = stack[index, AS_BACK]
            index -= 1
            continue
        elif (stack[index, AS_NMOVE] + 2) % 4 == stack[index, AS_PMOVE]:
            stack[index, AS_NMOVE] += 1
            continue

        # next position
        if stack[index, AS_NMOVE] == 0:
            ny, nx = (stack[index, AS_Y] - 1) % rows, stack[index, AS_X]
        elif stack[index, AS_NMOVE] == 1:
            ny, nx = stack[index, AS_Y], (stack[index, AS_X] + 1) % columns
        elif stack[index, AS_NMOVE] == 2:
            ny, nx = (stack[index, AS_Y] + 1) % rows, stack[index, AS_X]
        elif stack[index, AS_NMOVE] == 3:
            ny, nx = stack[index, AS_Y], (stack[index, AS_X] - 1) % columns

        # set next stack
        stack[index + 1] = [
            stack[index, 0] + 1,
            stack[index, 1] + 1,
            stack[index, 2] + 1,
            stack[index, 3] + 1,
            ny, nx, -1, stack[index, AS_NMOVE], 0]
        stack[index, AS_NMOVE] += 1

        # eating
        if field[FIELD_FOOD, stack[index, AS_Y], stack[index, AS_X]] == 1.0:
            stack[index + 1, player_index] -= 1

        # hungry turn
        if (step + index + 1) % 40 == 39:
            for oidx in [p for p in range(4) if p != player_index]:
                stack[index + 1, oidx] += 1

        if (step + index + 1) % 40 == 0:
            stack[index + 1, player_index] += 1

        index += 1

    return distance


class Agent(object):
    def __init__(self, model: nn.Module, config: Configuration, setting: Setting) -> None:
        self.config = config
        self.setting = setting
        self.worlds = [World(model, config, setting) for _ in range(MAX_SEARCH)]
        self.visit = 0
        self.postproc: Optional[Callable[['Agent', Observation, Action], None]] = None

    def __call__(self, obs: Observation, expand: bool = True) -> Action:
        if self.worlds[0].step != obs.step:
            self._search(obs, expand)

        moves = self.worlds[0].moves[obs.index]

        if (moves[:, MOVE_ENABLED] == 0.0).all():
            action_index = self.worlds[0].select_crush_action(obs.index)
        elif np.random.rand() < self.setting.random:
            action_probs = moves[:, 0] / (moves[:, MOVE_VISIT] + 1e-6)
            action_probs *= np.log(moves[:, MOVE_VISIT] + 1) * moves[:, MOVE_ENABLED]
            action_probs_sum = action_probs.sum()
            if action_probs_sum != 0:
                action_index = np.random.choice(4, p=action_probs / action_probs_sum)
            else:
                action_index = 0
        elif self.setting.decision == 'visit':
            action_index = (moves[:, MOVE_VISIT] * moves[:, MOVE_ENABLED]).argmax()
        elif self.setting.decision == 'value':
            player = len([v for v in self.worlds[0].lengths if v != 0])
            action_value = moves[:, 0] * (1.0 - self.setting.safe)
            action_value += moves[:, max(player - 2, 0)] * self.setting.safe
            action_value /= moves[:, MOVE_VISIT] + 1e-6
            action_index = (action_value * moves[:, MOVE_ENABLED]).argmax()
        else:  # hybrid
            player = len([v for v in self.worlds[0].lengths if v != 0])
            action_value = moves[:, 0] * (1.0 - self.setting.safe)
            action_value += moves[:, max(player - 2, 0)] * self.setting.safe
            action_value /= moves[:, MOVE_VISIT] + 1e-6
            action_value *= np.log(moves[:, MOVE_VISIT] + 1)
            action_index = (action_value * moves[:, MOVE_ENABLED]).argmax()

        action = [Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST][action_index]

        if self.postproc is not None:
            self.postproc(self, obs, action)

        if LOGGER.isEnabledFor(logging.DEBUG):
            for didx, direction in enumerate(['NORTH', ' EAST', 'SOUTH', ' WEST']):
                moves = self.worlds[0].moves[obs.index, didx]
                LOGGER.debug(
                    '[{}] {}{}: 1={:.2f}, 2={:.2f}, 3={:.2f}, v={:03d}, a={:.2f}, e={:d}, p={:.2f}'.format(
                        obs.index, '*' if didx == action_index else ' ', direction,
                        moves[0] / (moves[MOVE_VISIT] + 1e-6),
                        moves[1] / (moves[MOVE_VISIT] + 1e-6),
                        moves[2] / (moves[MOVE_VISIT] + 1e-6),
                        int(moves[MOVE_VISIT]), moves[MOVE_ALIVE],
                        int(moves[MOVE_ENABLED]), moves[MOVE_POLICY]))

        return action

    def _search(self, obs: Observation, expand: bool) -> None:
        # set time limit
        start_time = time.time()

        if not expand or self.setting.timelimit == 'zero':
            end_time = 0
            max_visit = MAX_SEARCH
        elif self.setting.timelimit == 'auto':
            remain_steps = self.config.episode_steps - obs.step
            remain_time = max(obs.remaining_overage_time - 5.0, 0.0)
            extra_time = remain_time / max(min(remain_steps, 150), 1)
            end_time = start_time + 0.98 + extra_time
            max_visit = MAX_SEARCH
        elif self.setting.timelimit == 'none':
            end_time = -1
            max_visit = MAX_SEARCH
        else:
            end_time = -1
            max_visit = min(int(self.setting.timelimit), MAX_SEARCH)

        # initial visit
        min_visit = 20 if expand and obs.step == 0 else 0

        # update the stage
        self.worlds[0].apply(obs, use_model=obs.step != 0, update_move=True)
        self.visit = 1

        # expand the tree until time limit
        while self.visit < max_visit:
            if end_time != -1 and self.visit >= min_visit:
                current_time = time.time()

                if self.visit < 5:
                    proc_time = 0.0
                else:
                    proc_time = (current_time - start_time) / self.visit

                if current_time + proc_time > end_time:
                    break

            self.worlds[0].expand(self.worlds[self.visit])
            self.visit += 1

        if LOGGER.isEnabledFor(logging.DEBUG):
            for key, val in sorted(obs.items(), key=lambda x: x[0]):
                LOGGER.debug('%s: %s', key, val)
            LOGGER.debug('visit: %d', self.visit)
            LOGGER.debug('elapsed: %.2f', time.time() - start_time)
            LOGGER.debug('length: %s', ', '.join(f'{v}' for v in self.worlds[0].lengths))
            LOGGER.debug('direction: %s', ', '.join(f'{v}' for v in self.worlds[0].directions))
            LOGGER.debug('value(1st): %s', ', '.join(f'{v:.2f}' for v in self.worlds[0].values[:, 0]))
            LOGGER.debug('value(2nd): %s', ', '.join(f'{v:.2f}' for v in self.worlds[0].values[:, 1]))
            LOGGER.debug('value(3rd): %s', ', '.join(f'{v:.2f}' for v in self.worlds[0].values[:, 2]))
            LOGGER.debug('\n%s', self.worlds[0])
# _END_AGENT_


def _print_input_data(indata):
    def position2char(a):
        if a[INPUT_FOOD] == 1.0:
            return ' F '

        for i, v in enumerate(a[:9]):
            if v == 0.0:
                continue
            elif i == 0:
                return '[m]'
            elif i == 1:
                return '<m>'
            elif i == 2:
                return '(m)'
            elif i == 3:
                return '|m|'
            elif i == 4:
                return ';m;'
            elif i == 5:
                return ':m:'
            elif i == 6:
                return ',m,'
            elif i == 7:
                return '.m.'
            elif i == 8:
                return ' m '

        for i, v in enumerate(a[9:39]):
            if v == 0.0:
                continue
            elif i % 10 == 0:
                return f'[{i // 10}]'
            elif i % 10 == 1:
                return f'<{i // 10}>'
            elif i % 10 == 2:
                return f'({i // 10})'
            elif i % 10 == 3:
                return f'|{i // 10}|'
            elif i % 10 == 4:
                return f';{i // 10};'
            elif i % 10 == 5:
                return f':{i // 10}:'
            elif i % 10 == 6:
                return f',{i // 10},'
            elif i % 10 == 7:
                return f'.{i // 10}.'
            elif i % 10 == 8:
                return f' {i // 10} '
            elif i % 10 == 9:
                return ' * '

        return ' . '

    def collision2char(a):
        for i, v in enumerate(a[INPUT_DIFFERENCE - 7:INPUT_DIFFERENCE + 8]):
            if v != 0.0:
                return f'{i-7:+d} '

        if a[INPUT_COLLISION] != 0.0:
            return ' # '
        elif a[INPUT_COLLISION + 1] != 0.0:
            return ' * '
        else:
            return ' . '

    print('positions:')
    for y in range(INPUT_AREA_SIZE):
        print(''.join(
            position2char(indata[:, y, x])
            for x in range(INPUT_AREA_SIZE)), flush=True)

    print('collisions:')
    for y in range(INPUT_AREA_SIZE):
        print(''.join(
            collision2char(indata[:, y, x])
            for x in range(INPUT_AREA_SIZE)), flush=True)

    print('hungry:', indata[INPUT_HUNGRY:INPUT_HUNGRY + 8, 0, 0])
    print('end:', indata[INPUT_END:INPUT_END + 8, 0, 0])
    print('step:', indata[INPUT_STEP:INPUT_STEP + 10, 0, 0])
    print('direction:', indata[INPUT_DIRECTION:INPUT_DIRECTION + 2, 0, 0])
    print('players:', indata[INPUT_PLAYERS:INPUT_PLAYERS + 3, 0, 0])
