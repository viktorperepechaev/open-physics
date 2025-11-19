from dataclasses import dataclass, field
from typing import Tuple, Optional
from enum import Enum

class InteractionType(Enum):
    IMPULSE = "impulse"
    HOOKE = "hooke"

@dataclass
class BallPreset:
    """
    Настройка конкретного шарика.

    Если какое-то поле = None, то оно будет сгенерировано случайно:
    - radius: радиус шарика
    - speed: модуль скорости
    - angle_deg: направление скорости в градусах (0° — вправо, 90° — вверх)
    - position: начальная позиция центра (x, y) в пикселях
    - color: цвет (R, G, B)
    """
    radius: Optional[float] = None
    speed: Optional[float] = None
    angle_deg: Optional[float] = None
    position: Optional[Tuple[float, float]] = None
    color: Optional[Tuple[int, int, int]] = None
    mass: Optional[float] = None


@dataclass
class SimulationConfig:
    width: int = 800
    height: int = 600

    ball_radius: float = 20.0
    ball_color: Tuple[int, int, int] = (180, 220, 255)
    ball_mass: float = 1.0

    min_initial_speed: float = 150.0
    max_initial_speed: float = 250.0

    restitution: float = 1.0

    background_color: Tuple[int, int, int] = (10, 10, 20)
    fps: int = 60

    default_num_balls: int = 10

    ball_presets: list[BallPreset] = field(default_factory=list)

    interaction_type: InteractionType = InteractionType.HOOKE

    hooke_k_ball: float = 500.0
    hooke_k_wall: float = 500.0


CONFIG = SimulationConfig(
    ball_presets=[
         BallPreset(
             radius=100.0,
             speed=100.0,
             angle_deg=90.0,
             position=(100.0, 300.0),
             color=(255, 100, 100),
             mass=30.0,
         ),

         BallPreset(
             radius=25.0,
             speed=0.0,
             angle_deg=0.0, 
             position=(400.0, 300.0),
             color=(100, 255, 100),
             mass=5.0
         ),
    ]
)

