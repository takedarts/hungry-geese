from typing import List

import torch.jit
import torch.nn as nn

from config import (INPUT_AREA_SIZE, INPUT_CHANNELS, NORMAL_MODEL_BLOCKS,
                    NORMAL_MODEL_CHANNELS, SMALL_MODEL_BLOCKS,
                    SMALL_MODEL_CHANNELS, TINY_MODEL_BLOCKS,
                    TINY_MODEL_CHANNELS)

# stem type
# 0:11x11->11x11
# 1:11x11->9x9
# 2:11x11->7x7
# 3:11x11->5x5
# 4:11x11->3x3
STEM_TYPE = 2

# head type
# 1: fc-fc
# 2: conv-fc
HEAD_TYPE = 1

if STEM_TYPE == 0:
    BLOCK_SIZE = INPUT_AREA_SIZE
    BLOCK_DEPTH = 0.50
elif STEM_TYPE == 1:
    BLOCK_SIZE = INPUT_AREA_SIZE - 2
    BLOCK_DEPTH = 0.75
elif STEM_TYPE == 2:
    BLOCK_SIZE = INPUT_AREA_SIZE - 4
    BLOCK_DEPTH = 1.0
elif STEM_TYPE == 3:
    BLOCK_SIZE = INPUT_AREA_SIZE - 6
    BLOCK_DEPTH = 1.64
elif STEM_TYPE == 4:
    BLOCK_SIZE = INPUT_AREA_SIZE - 8
    BLOCK_DEPTH = 2.42
else:
    raise Exception(f'unsupported stem type: {STEM_TYPE}')


class Reshape(nn.Module):

    def __init__(self, *shape) -> None:
        super().__init__()

        if len(shape) == 1 and hasattr(shape[0], '__getitem__'):
            self.shape = tuple(shape[0])
        else:
            self.shape = tuple(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], *self.shape)

    def extra_repr(self) -> str:
        return str(self.shape)


class PositionalEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(BLOCK_SIZE ** 2, channels)
        self.register_buffer('pos', torch.arange(BLOCK_SIZE ** 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(self.pos)
        e = e.permute(1, 0).reshape(1, -1, BLOCK_SIZE, BLOCK_SIZE)
        return x + e


class Stem(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        padding = 0 if STEM_TYPE > 0 else 1
        modules: List[nn.Module] = [nn.Conv2d(
            INPUT_CHANNELS, channels, kernel_size=3, padding=padding, bias=False),
        ]

        for _ in range(1, STEM_TYPE):
            modules.extend([
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            ])

        modules.append(nn.BatchNorm2d(channels))

        self.op = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            PositionalEmbedding(channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.op(x)


class Head(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.act = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        if HEAD_TYPE == 2:
            policy_head = [
                nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                Reshape(-1),
                nn.Linear((BLOCK_SIZE ** 2) * channels, channels, bias=False),
            ]
            value_head = [
                nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                Reshape(-1),
                nn.Linear(channels, channels, bias=False),
            ]
        else:
            policy_head = [
                Reshape(-1),
                nn.Linear((BLOCK_SIZE ** 2) * channels, channels, bias=False),
                nn.LayerNorm(channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, channels, bias=False),
            ]
            value_head = [
                Reshape(-1),
                nn.Linear((BLOCK_SIZE ** 2) * channels, channels, bias=False),
                nn.LayerNorm(channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, channels, bias=False),
            ]

        policy_head.extend([
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 3, bias=True),
        ])

        value_head.extend([
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 3, bias=True),
        ])

        self.policy = nn.Sequential(*policy_head)
        self.value = nn.Sequential(*value_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(x)
        p = self.policy(x)
        v = self.value(x)

        return torch.cat([p, v], dim=1)


class Model(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        self.stem = Stem(channels)
        self.blocks = nn.Sequential(
            *[Block(channels) for _ in range(round(blocks * BLOCK_DEPTH))])
        self.head = Head(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x.sigmoid()


class NormalModel(Model):
    def __init__(self) -> None:
        super().__init__(NORMAL_MODEL_CHANNELS, NORMAL_MODEL_BLOCKS)


class SmallModel(Model):
    def __init__(self) -> None:
        super().__init__(SMALL_MODEL_CHANNELS, SMALL_MODEL_BLOCKS)


class TinyModel(Model):
    def __init__(self) -> None:
        super().__init__(TINY_MODEL_CHANNELS, TINY_MODEL_BLOCKS)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones([x.shape[0], 6], dtype=torch.float)
