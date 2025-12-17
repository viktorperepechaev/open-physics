import numpy as np
from scipy.integrate import solve_ivp

from objects import Ball, Table
from config import CONFIG, InteractionType


def ball_ode(t: float, y: np.ndarray) -> np.ndarray:
    """
    ОДУ для идеально упругого столкновения.
    y = [x, y, vx, vy]
    """
    dxdt = y[2]
    dydt = y[3]
    dvxdt = 0.0
    dvydt = 0.0
    return np.array([dxdt, dydt, dvxdt, dvydt], dtype=float)


def reflect_velocity(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    vn = float(np.dot(v, n))
    return v - 2.0 * vn * n


def integrate_ball_step(ball: Ball, table: Table, dt: float) -> None:
    y0 = ball.state_vector()

    sol = solve_ivp(
        ball_ode,
        t_span=(0.0, dt),
        y0=y0,
        t_eval=[dt],
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        return

    y_end = sol.y[:, -1]
    ball.set_from_state(y_end)

    x, y = ball.position
    r = ball.radius
    W, H = table.width, table.height

    # Левая стенка
    if x - r < 0.0:
        ball.position[0] = r
        ball.velocity = reflect_velocity(ball.velocity, np.array([1.0, 0.0]))

    # Правая стенка
    if x + r > W:
        ball.position[0] = W - r
        ball.velocity = reflect_velocity(ball.velocity, np.array([-1.0, 0.0]))

    # Верхняя стенка
    if y - r < 0.0:
        ball.position[1] = r
        ball.velocity = reflect_velocity(ball.velocity, np.array([0.0, 1.0]))

    # Нижняя стенка
    if y + r > H:
        ball.position[1] = H - r
        ball.velocity = reflect_velocity(ball.velocity, np.array([0.0, -1.0]))

def handle_ball_collisions(balls: list[Ball]) -> None:
    """
    Обрабатываем идеально упругое столкновение между произвольными шарами.
    """
    n_balls = len(balls)
    if n_balls < 2:
        return

    for i in range(n_balls):
        for j in range(i + 1, n_balls):
            b1 = balls[i]
            b2 = balls[j]

            delta = b2.position - b1.position
            dist = float(np.linalg.norm(delta))
            min_dist = b1.radius + b2.radius

            if dist >= min_dist:
                continue

            n = delta / dist

            overlap = min_dist - dist
            m1, m2 = b1.mass, b2.mass
            total_mass = m1 + m2 if (m1 + m2) != 0.0 else 1.0

            b1.position -= (overlap * (m2 / total_mass)) * n
            b2.position += (overlap * (m1 / total_mass)) * n

            v1 = b1.velocity
            v2 = b2.velocity
            rel_v = v1 - v2
            u_n = float(np.dot(rel_v, n))

            if u_n <= 0.0:
                continue

            factor1 = 2.0 * m2 / (m1 + m2)
            factor2 = 2.0 * m1 / (m1 + m2)

            b1.velocity = v1 - factor1 * u_n * n
            b2.velocity = v2 + factor2 * u_n * n

def system_ode_hooke(
    t: float,
    y: np.ndarray,
    radii: np.ndarray,
    masses: np.ndarray,
    table: Table,
    k_ball: float,
    k_wall: float,
) -> np.ndarray:
    """
    ОДУ для модели с силами по закону Гука.
    y = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
    """
    n_balls = masses.shape[0]
    state = y.reshape(n_balls, 4)

    pos = state[:, 0:2]
    vel = state[:, 2:4]
    acc = np.zeros_like(pos)

    for i in range(n_balls):
        for j in range(i + 1, n_balls):
            r1 = pos[i]
            r2 = pos[j]
            delta = r2 - r1
            dist = float(np.linalg.norm(delta))
            min_dist = radii[i] + radii[j]
            overlap = min_dist - dist

            if overlap <= 0.0 or dist == 0.0:
                continue

            n = delta / dist

            F12 = -k_ball * overlap * n

            acc[i] += F12 / masses[i]
            acc[j] -= F12 / masses[j]

    W, H = table.width, table.height
    for i in range(n_balls):
        x, y_coord = pos[i]
        r = radii[i]
        m = masses[i]

        # левая стенка
        overlap = r - x
        if overlap > 0.0:
            acc[i] += np.array([k_wall * overlap, 0.0]) / m

        # правая стенка
        overlap = x + r - W
        if overlap > 0.0:
            acc[i] += np.array([-k_wall * overlap, 0.0]) / m

        # верхняя стенка
        overlap = r - y_coord
        if overlap > 0.0:
            acc[i] += np.array([0.0, k_wall * overlap]) / m

        # нижняя стенка
        overlap = y_coord + r - H
        if overlap > 0.0:
            acc[i] += np.array([0.0, -k_wall * overlap]) / m

    dxdt = vel
    dvdt = acc
    return np.hstack([dxdt, dvdt]).reshape(-1)

def integrate_system_step_hooke(balls: list[Ball], table: Table, dt: float) -> None:
    if not balls:
        return

    y0_parts = [ball.state_vector() for ball in balls]
    y0 = np.concatenate(y0_parts)

    radii = np.array([b.radius for b in balls], dtype=float)
    masses = np.array([b.mass for b in balls], dtype=float)

    sol = solve_ivp(
        system_ode_hooke,
        t_span=(0.0, dt),
        y0=y0,
        t_eval=[dt],
        args=(radii, masses, table, CONFIG.hooke_k_ball, CONFIG.hooke_k_wall),
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        return

    y_end = sol.y[:, -1]

    for i, ball in enumerate(balls):
        state_i = y_end[4 * i:4 * i + 4]
        ball.set_from_state(state_i)

def step_physics(balls: list[Ball], table: Table, dt: float) -> None:
    if CONFIG.interaction_type == InteractionType.HOOKE:
        integrate_system_step_hooke(balls, table, dt)
    else:
        for ball in balls:
            integrate_ball_step(ball, table, dt)
        handle_ball_collisions(balls)

