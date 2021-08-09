import base64
import io
import itertools
import logging
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch.jit
import torch.nn as nn
from kaggle_environments.envs.hungry_geese.hungry_geese \
    import Action, Configuration, Observation
from numba import njit

torch.set_num_threads(1)

LOGGER = logging.getLogger(__name__)

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3
INVALID = -1

INPUT_AREA_SIZE = 11
INPUT_CHANNELS = 98

MAX_SEARCH = 160

SETTING_TIMELIMIT = 'auto'
SETTING_DEPTHLIMIT = 20
SETTING_DECISION = 'hybrid'
SETTING_NORMALIZE = True
SETTING_COLLISION = 0.0
SETTING_EATING = 0.0
SETTING_SEARCH = 'ucbm'
SETTING_SAFE = 0.0
SETTING_STRICT = False
SETTING_LOOKAHEAD = 4
SETTING_POLICYBASE = 0.1
SETTING_VALUEBASE = 0.1
SETTING_VALUETAIL = 0.05
SETTING_USEPOLICY = True
SETTING_USEVALUE = True

CACHED_AGENTS: Dict[int, 'Agent'] = {}
MODEL: Optional[nn.Module] = None
MODEL_BASE64: Optional[str] = None

VISIT = 6
ALIVE = 7
ENABLED = 8
POLICY = 9
EXPANDED = 10

# _BEGIN_AGENT_ #


class Setting(object):
    pass


class Agent(object):
    visit: int
    worlds = []  # type:ignore


# _END_AGENT_ #


def postprocess(agent: Agent, obs: Observation, action: Action):
    global MODEL

    print('visit: {}, thread={}, remain={:.2f}'.format(
        agent.visit, torch.get_num_threads(), float(obs.remaining_overage_time)))
    print('actions:')
    for didx, direction in enumerate(['NORTH', 'EAST', 'SOUTH', 'WEST']):
        moves = agent.worlds[0].moves[obs.index, didx]
        print(' {}{:>5s}: 1={:.2f}, 2={:.2f}, 3={:.2f}, v={:03d}, a={:.2f}, e={:d}, p={:.2f}'.format(
            '*' if direction == action.name else ' ', direction,
            moves[0] / (moves[VISIT] + 1e-6),
            moves[1] / (moves[VISIT] + 1e-6),
            moves[2] / (moves[VISIT] + 1e-6),
            int(moves[VISIT]), moves[ALIVE],
            int(moves[ENABLED]), moves[POLICY]))


def agent(obs, config):
    global CACHED_AGENTS, MODEL, MODEL_BASE64_0, MODEL_BASE64_1
    global SETTING_TIMELIMIT, SETTING_DEPTHLIMIT, SETTING_DECISION, SETTING_NORMALIZE
    global SETTING_COLLISION, SETTING_EATING, SETTING_SEARCH, SETTING_SAFE, SETTING_STRICT
    global SETTING_LOOKAHEAD, SETTING_POLICYBASE, SETTING_VALUEBASE, SETTING_VALUETAIL
    global SETTING_USEPOLICY, SETTING_USEVALUE
    observation = Observation(obs)

    if MODEL is None:
        MODEL = torch.jit.load(
            io.BytesIO(base64.b64decode(MODEL_BASE64.encode('utf-8'))))

    if observation.index not in CACHED_AGENTS:
        setting = Setting(
            timelimit=SETTING_TIMELIMIT,
            depthlimit=SETTING_DEPTHLIMIT,
            decision=SETTING_DECISION,
            normalize=SETTING_NORMALIZE,
            collision=SETTING_COLLISION,
            eating=SETTING_EATING,
            search=SETTING_SEARCH,
            safe=SETTING_SAFE,
            strict=SETTING_STRICT,
            lookahead=SETTING_LOOKAHEAD,
            policybase=SETTING_POLICYBASE,
            valuebase=SETTING_VALUEBASE,
            valuetail=SETTING_VALUETAIL,
            usepolicy=SETTING_USEPOLICY,
            usevalue=SETTING_USEVALUE,
        )

        CACHED_AGENTS[observation.index] = Agent(MODEL, Configuration(config), setting)
        CACHED_AGENTS[observation.index].postproc = postprocess

    return CACHED_AGENTS[observation.index](observation).name


# _MODEL_NAME_
# _MODEL_BASE64_
