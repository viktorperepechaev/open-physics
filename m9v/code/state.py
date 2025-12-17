from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from config import CONFIG


@dataclass
class CAState:
    temp: np.ndarray  # (nx, ny)
    solid: np.ndarray  # (nx, ny)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.temp.shape

    @property
    def nx(self) -> int:
        return self.temp.shape[0]

    @property
    def ny(self) -> int:
        return self.temp.shape[1]


def create_initial_state(cfg=CONFIG) -> CAState:
    nx, ny = cfg.nx, cfg.ny

    temp = np.full((nx, ny), cfg.T_liquid_init, dtype=float)
    solid = np.zeros((nx, ny), dtype=bool)

    if cfg.seed_solid_top_edge:
        solid[0, :] = True
        temp[0, :] = cfg.T_solid_seed

    if cfg.random_seed_fraction > 0:
        rng = np.random.default_rng(cfg.rng_seed)
        
        random_mask = rng.random((nx, ny)) < cfg.random_seed_fraction
        
        solid = np.logical_or(solid, random_mask)
        
        temp[solid] = cfg.T_solid_seed

    return CAState(temp=temp, solid=solid)

