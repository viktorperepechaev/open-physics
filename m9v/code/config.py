from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Neighbourhood(Enum):
    VON_NEUMANN = "von_neumann" # 4 соседа
    MOORE = "moore" # 8 соседей


@dataclass
class SimulationConfig:
    # Размер решётки
    nx: int = 80
    ny: int = 80

    dt: float = 0.1
    total_steps: int = 5000  # максимум шагов модели

    # Температуры
    T_melt: float = 0.0 # T_плавления
    T_liquid_init: float = -5.0 # начальная температура воды
    T_solid_seed: float = -5.0 # температура льда

    # Теплопроводность и теплоёмкость
    alpha: float = 0.15 # dt / C
    C: float = 1.0 # теплоёмкость ячейки
    L: float = 5.0 # теплота плавления на ячейку

    # Вероятность замерзания
    beta: float = 0.1 # чувствительность к переохлаждению

    neighbourhood: Neighbourhood = Neighbourhood.VON_NEUMANN

    seed_solid_top_edge: bool = True  # заморозить верхнюю строку как лёд

    rng_seed: int = 42

    print_every: int = 200

    cell_size: int = 8 # размер клетки в пикселях
    fps: int = 60 # целевой FPS
    ca_steps_per_frame: int = 10  # сколько шагов модели делать за один кадр

    color_background: Tuple[int, int, int] = (5, 5, 15)

    color_cold_water: Tuple[int, int, int] = (10, 40, 120)
    color_hot_water: Tuple[int, int, int] = (60, 160, 255)

    # Лёд
    color_ice: Tuple[int, int, int] = (230, 240, 255)

    vis_temp_span: float = 10.0


CONFIG = SimulationConfig()

