from __future__ import annotations

import sys

import numpy as np
import pygame

from config import CONFIG
from state import create_initial_state, CAState
from rules import step as ca_step


def compute_solid_fraction(solid: np.ndarray) -> float:
    """Доля твёрдых ячеек в решётке."""
    total = solid.size
    if total == 0:
        return 0.0
    return float(solid.sum()) / float(total)


def lerp(a: float, b: float, t: float) -> float:
    """Линейная интерполяция между a и b."""
    return a + (b - a) * t


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def color_for_cell(
    T: np.ndarray,
    solid: np.ndarray,
    i: int,
    j: int,
    cfg=CONFIG,
) -> tuple[int, int, int]:
    """
    - лёд -> cfg.color_ice
    - вода -> градиент от cfg.color_cold_water до cfg.color_hot_water
              в зависимости от T относительно T_melt.
    """
    if solid[i, j]:
        return cfg.color_ice

    temp = T[i, j]

    if temp <= cfg.T_melt:
        t = 0.0
    else:
        t = (temp - cfg.T_melt) / cfg.vis_temp_span
    t = clamp01(t)

    c0 = cfg.color_cold_water
    c1 = cfg.color_hot_water
    r = int(lerp(c0[0], c1[0], t))
    g = int(lerp(c0[1], c1[1], t))
    b = int(lerp(c0[2], c1[2], t))
    return r, g, b


def draw_state(
    screen: pygame.Surface,
    state: CAState,
    cfg=CONFIG,
) -> None:
    nx, ny = state.shape
    cs = cfg.cell_size

    screen.fill(cfg.color_background)

    for i in range(nx):
        y = i * cs
        for j in range(ny):
            x = j * cs
            color = color_for_cell(state.temp, state.solid, i, j, cfg)
            rect = pygame.Rect(x, y, cs, cs)
            pygame.draw.rect(screen, color, rect)


def run_pygame_simulation() -> None:
    cfg = CONFIG
    rng = np.random.default_rng(cfg.rng_seed)

    state = create_initial_state(cfg)
    nx, ny = state.shape

    pygame.init()
    pygame.display.set_caption("Freezing Cellular Automaton (Water -> Ice)")
    width = ny * cfg.cell_size
    height = nx * cfg.cell_size
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    print(f"Grid: {nx} x {ny}")
    print(f"T_melt = {cfg.T_melt}, alpha = {cfg.alpha}, beta = {cfg.beta}")
    print(f"L = {cfg.L}, C = {cfg.C}")
    print(f"Neighbourhood = {cfg.neighbourhood.value}")
    print(f"Initial solid fraction = {compute_solid_fraction(state.solid):.3f}")
    print("SPACE — пауза/продолжить, ESC — выход.\n")

    running = True
    paused = False
    step_idx = 0
    simulation_finished = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'} at step {step_idx}")

        if not paused and not simulation_finished:
            for _ in range(cfg.ca_steps_per_frame):
                if step_idx >= cfg.total_steps:
                    simulation_finished = True
                    print(f"Reached total_steps = {cfg.total_steps}, stopping updates.")
                    break

                state = ca_step(state, cfg=cfg, rng=rng)
                step_idx += 1

                frac = compute_solid_fraction(state.solid)
                if (step_idx % cfg.print_every) == 0:
                    print(
                        f"step = {step_idx:6d} | solid fraction = {frac:.3f}"
                    )

                if frac >= 0.999:
                    simulation_finished = True
                    print(
                        f"All cells are (almost) solid at step {step_idx}. "
                        "Stopping updates."
                    )
                    break

        draw_state(screen, state, cfg)
        pygame.display.flip()

        clock.tick(cfg.fps)

    pygame.quit()


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    run_pygame_simulation()


if __name__ == "__main__":
    main()

