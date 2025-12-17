import math
import random

import pygame
import numpy as np

from config import CONFIG, BallPreset
from objects import Ball, Table
from physics import step_physics


def create_ball(table: Table, preset: BallPreset | None = None) -> Ball:
    if preset is not None and preset.radius is not None:
        r = float(preset.radius)
    else:
        r = CONFIG.ball_radius

    if preset is not None and preset.position is not None:
        x, y = preset.position
    else:
        x = random.uniform(r, table.width - r)
        y = random.uniform(r, table.height - r)

    if preset is not None and preset.speed is not None:
        speed = float(preset.speed)
    else:
        speed = random.uniform(CONFIG.min_initial_speed,
                               CONFIG.max_initial_speed)

    if preset is not None and preset.angle_deg is not None:
        angle = math.radians(float(preset.angle_deg))
    else:
        angle = random.uniform(0.0, 2.0 * math.pi)

    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)

    if preset is not None and preset.color is not None:
        color = preset.color
    else:
        color = CONFIG.ball_color

    if preset is not None and preset.mass is not None:
        mass = float(preset.mass)
    else:
        mass = CONFIG.ball_mass

    return Ball(
        position=np.array([x, y], dtype=float),
        velocity=np.array([vx, vy], dtype=float),
        radius=r,
        color=color,
        mass=mass,
    )


def main() -> None:
    num_balls = max(1, CONFIG.default_num_balls)

    pygame.init()
    screen = pygame.display.set_mode((CONFIG.width, CONFIG.height))

    clock = pygame.time.Clock()

    table = Table(width=CONFIG.width, height=CONFIG.height)

    font = pygame.font.Font(None, 24)

    balls: list[Ball] = []
    presets = CONFIG.ball_presets

    for i in range(num_balls):
        preset = presets[i] if i < len(presets) else None
        balls.append(create_ball(table, preset))

    running = True

    sim_time = 0.0
    while running:
        dt = clock.tick(CONFIG.fps) / 1000.0

        sim_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        step_physics(balls, table, dt)

        screen.fill(CONFIG.background_color)

        for ball in balls:
            pygame.draw.circle(
                screen,
                ball.color,
                (int(ball.position[0]), int(ball.position[1])),
                int(ball.radius),
            )

        all_energy = 0.0
        for ball in balls:
            v2 = float(ball.velocity[0] ** 2 + ball.velocity[1] ** 2)
            all_energy += 0.5 * ball.mass * v2

        time_surface = font.render(
            f"t = {sim_time:.2f} s, E = {all_energy:.2f}",
            True,
            (255, 255, 255),
        )

        screen.blit(time_surface, (10, 10))

        pygame.display.flip()



    pygame.quit()


if __name__ == "__main__":
    main()

