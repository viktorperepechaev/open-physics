from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Table:
    width: int
    height: int


@dataclass
class Ball:
    position: np.ndarray # [x, y]
    velocity: np.ndarray # [vx, vy]
    radius: float
    color: Tuple[int, int, int]
    mass: float

    def state_vector(self) -> np.ndarray:
        return np.array([self.position[0], self.position[1],
                         self.velocity[0], self.velocity[1]],
                        dtype=float)

    def set_from_state(self, y: np.ndarray) -> None:
        self.position = np.array([y[0], y[1]], dtype=float)
        self.velocity = np.array([y[2], y[3]], dtype=float)

