from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from config import CONFIG, Neighbourhood
from state import CAState


def neighbour_offsets(neighbourhood: Neighbourhood) -> Tuple[Tuple[int, int], ...]:
    if neighbourhood == Neighbourhood.VON_NEUMANN:
        return (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        )
    elif neighbourhood == Neighbourhood.MOORE:
        return (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
    else:
        raise ValueError(f"Unknown neighbourhood: {neighbourhood}")


def iter_neighbours(
    i: int,
    j: int,
    nx: int,
    ny: int,
    neighbourhood: Neighbourhood,
) -> Iterator[Tuple[int, int]]:
    for di, dj in neighbour_offsets(neighbourhood):
        ni = i + di
        nj = j + dj
        if 0 <= ni < nx and 0 <= nj < ny:
            yield ni, nj


def diffusion_step(
    state: CAState,
    alpha: float,
    neighbourhood: Neighbourhood,
) -> np.ndarray:
    nx, ny = state.shape
    T_old = state.temp
    T_new = np.empty_like(T_old)

    offsets = neighbour_offsets(neighbourhood)

    for i in range(nx):
        for j in range(ny):
            T_ij = T_old[i, j]

            delta_sum = 0.0
            for di, dj in offsets:
                ni = i + di
                nj = j + dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    delta_sum += T_old[ni, nj] - T_ij

            T_new[i, j] = T_ij + alpha * delta_sum

    return T_new


def count_solid_neighbours(
    solid: np.ndarray,
    i: int,
    j: int,
    neighbourhood: Neighbourhood,
) -> Tuple[int, int]:
    nx, ny = solid.shape
    n_solid = 0
    n_max = 0

    for ni, nj in iter_neighbours(i, j, nx, ny, neighbourhood):
        n_max += 1
        if solid[ni, nj]:
            n_solid += 1

    return n_solid, n_max


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def freezing_step(
    T: np.ndarray,
    solid: np.ndarray,
    cfg=CONFIG,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    nx, ny = solid.shape
    new_solid = solid.copy()
    frozen_this_step = np.zeros_like(solid, dtype=bool)

    for i in range(nx):
        for j in range(ny):
            if solid[i, j]:
                continue

            T_ij = T[i, j]
            if T_ij >= cfg.T_melt:
                continue

            delta_T = cfg.T_melt - T_ij  # > 0
            n_solid, n_max = count_solid_neighbours(solid, i, j, cfg.neighbourhood)

            if n_max == 0:
                continue

            if n_solid == 0:
                continue

            p = cfg.beta * delta_T * (n_solid / n_max)

            if n_solid == n_max:
                p = 1.0
            else:
                p = clamp01(p)

            u = rng.random()
            if u < p:
                new_solid[i, j] = True
                frozen_this_step[i, j] = True

    return new_solid, frozen_this_step


def latent_heat_step(
    T: np.ndarray,
    frozen_this_step: np.ndarray,
    cfg=CONFIG,
) -> np.ndarray:
    nx, ny = T.shape
    T_new = T.copy()

    for i in range(nx):
        for j in range(ny):
            if not frozen_this_step[i, j]:
                continue

            neighbours = list(
                iter_neighbours(i, j, nx, ny, cfg.neighbourhood)
            )
            if not neighbours:
                continue

            delta_T_neighbor = cfg.L / (cfg.C * len(neighbours))

            for ni, nj in neighbours:
                #T_new[ni, nj] -= delta_T_neighbor
                T_new[ni, nj] += delta_T_neighbor

    return T_new


def step(
    state: CAState,
    cfg=CONFIG,
    rng: np.random.Generator | None = None,
) -> CAState:
    if rng is None:
        rng = np.random.default_rng()

    T_after_diff = diffusion_step(
        state=state,
        alpha=cfg.alpha,
        neighbourhood=cfg.neighbourhood,
    )

    new_solid, frozen_this_step = freezing_step(
        T=T_after_diff,
        solid=state.solid,
        cfg=cfg,
        rng=rng,
    )

    T_after_latent = latent_heat_step(
        T=T_after_diff,
        frozen_this_step=frozen_this_step,
        cfg=cfg,
    )

    return CAState(temp=T_after_latent, solid=new_solid)

