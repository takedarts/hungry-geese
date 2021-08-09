'''This script makes a submission file for the hungry-geese competition.
The specified model is encoded to base64 string and embedded in the submission file.
The submission file is created in the data directory (data/submission.py).
'''
import argparse
import base64
import io
import logging
import pathlib
from typing import List

import torch.jit

from utils import setup_logging

parser = argparse.ArgumentParser(
    description='Make a submission file',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('model', type=str, help='Model file name.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')

LOGGER = logging.getLogger(__name__)


def make_model_code(filename: str) -> List[str]:
    LOGGER.debug('read inference model: %s', filename)
    model = torch.jit.load(filename)

    buffer = io.BytesIO()
    torch.jit.save(model, buffer)
    text = base64.b64encode(buffer.getbuffer())
    model_code = ['MODEL_BASE64 = """']
    model_code += [text[i:i + 100].decode('utf-8') for i in range(0, len(text), 100)]
    model_code += ['"""', '']

    return model_code


def main():
    args = parser.parse_args()
    setup_logging(args.debug)

    agent_path = pathlib.Path(__file__).parent / 'agent.py'
    submission_path = pathlib.Path(__file__).parent / 'submission.py'
    output_path = pathlib.Path(__file__).parent.parent / 'data' / 'submission.py'

    # read agent code
    agent_code = []

    with open(agent_path, 'r') as reader:
        phase = 0
        for line in reader:
            if phase == 0 and '# _BEGIN_AGENT_' in line:
                phase = 1
            elif '# _END_AGENT_' in line:
                phase = 0
            elif phase == 1:
                agent_code.append(line)

    # model data
    model_name = 'MODEL_NAME = \'{}\''.format(args.model)
    model_code = make_model_code(args.model)

    # write submission file
    LOGGER.debug('write submission file: %s', output_path)
    with open(submission_path, 'r') as reader, open(output_path, 'w') as writer:
        phase = 0
        for line in reader:
            if phase == 0 and '# _BEGIN_AGENT_' in line:
                phase = 1
                writer.write(''.join(agent_code))
            elif '# _END_AGENT_' in line:
                phase = 0
            elif '# _MODEL_NAME_' in line:
                writer.write(f'{model_name}\n')
            elif '# _MODEL_BASE64_' in line:
                writer.write('\n'.join(model_code))
            elif phase == 0:
                writer.write(line)


if __name__ == '__main__':
    main()
