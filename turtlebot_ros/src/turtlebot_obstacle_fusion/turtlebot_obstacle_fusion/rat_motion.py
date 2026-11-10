"""RatInABox 2D random-motion model ported for a robot that perceives walls as line segments.

The update follows ``ratinabox.Agent.update`` (commit d8b85d5): an Ornstein-Uhlenbeck (OU)
process on the rotational velocity, an OU process on the speed after mapping it from a
Rayleigh to a standard normal variable, a spring-like wall repulsion acting on the velocity
and a conveyor-belt-like wall repulsion acting only on the displacement.

All vectors are expressed in a frame centred on the robot whose axes are aligned with the
odometry frame, so the robot is always at the origin and segments are relative to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, log, pi, sin, sqrt
from statistics import NormalDist
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

_STANDARD_NORMAL = NormalDist()
_EPS = 1e-6

# Unscaled rat parameters from simulation/config.yaml. ROT_SPEED_STD is in units of pi rad/s.
BEHAVIOURS: Dict[str, Dict[str, float]] = {
    "crawl": {"SPEED_MEAN": 0.03794, "SPEED_CT": 2.0, "ROT_SPEED_STD": 0.25, "ROT_SPEED_CT": 0.075, "THIGMOTAXIS": 0.6},
    "walk": {"SPEED_MEAN": 0.09087, "SPEED_CT": 1.35, "ROT_SPEED_STD": 0.667, "ROT_SPEED_CT": 0.075, "THIGMOTAXIS": 0.6},
    "run": {"SPEED_MEAN": 0.07081, "SPEED_CT": 1.35, "ROT_SPEED_STD": 0.667, "ROT_SPEED_CT": 0.075, "THIGMOTAXIS": 0.6},
    "adult": {"SPEED_MEAN": 0.08021, "SPEED_CT": 1.35, "ROT_SPEED_STD": 0.8334, "ROT_SPEED_CT": 0.075, "THIGMOTAXIS": 0.4},
}
RAT_WALL_REPEL_DISTANCE = 0.1


@dataclass(frozen=True)
class MotionParams:
    """RatInABox parameters already converted to robot length and time units."""

    speed_mean: float
    speed_coherence_time: float
    rotational_velocity_std: float
    rotational_velocity_coherence_time: float
    thigmotaxis: float
    wall_repel_distance: float
    wall_repel_strength: float = 1.0


def scaled_params(behaviour: str, length_scale: float, time_scale: float) -> MotionParams:
    """Convert rat parameters with lengths multiplied by ``length_scale`` and times by ``time_scale``."""
    raw = BEHAVIOURS[behaviour]
    return MotionParams(
        speed_mean=raw["SPEED_MEAN"] * length_scale / time_scale,
        speed_coherence_time=raw["SPEED_CT"] * time_scale,
        rotational_velocity_std=raw["ROT_SPEED_STD"] * pi / time_scale,
        rotational_velocity_coherence_time=raw["ROT_SPEED_CT"] * time_scale,
        thigmotaxis=raw["THIGMOTAXIS"],
        wall_repel_distance=RAT_WALL_REPEL_DISTANCE * length_scale,
    )


def ornstein_uhlenbeck(dt: float, x: float, drift: float, noise_scale: float, coherence_time: float, rng: np.random.Generator) -> float:
    """Return the OU increment exactly as ``ratinabox.utils.ornstein_uhlenbeck``."""
    sigma = sqrt(2.0 * noise_scale ** 2 / (coherence_time * dt))
    return (drift - x) * dt / coherence_time + sigma * rng.normal(scale=dt)


def rayleigh_to_normal(speed: float, sigma: float) -> float:
    """Map a Rayleigh(sigma) distributed speed to a standard normal variable."""
    uniform = 1.0 - exp(-(speed ** 2) / (2.0 * sigma ** 2))
    return _STANDARD_NORMAL.inv_cdf(min(max(_EPS, uniform), 1.0 - _EPS))


def normal_to_rayleigh(z: float, sigma: float) -> float:
    """Map a standard normal variable to a Rayleigh(sigma) distributed speed."""
    uniform = min(_STANDARD_NORMAL.cdf(z), 1.0 - 1e-12)
    return sigma * sqrt(-2.0 * log(1.0 - uniform))


def rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D vector anticlockwise by ``angle``."""
    c, s = cos(angle), sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def wall_bounce(velocity: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Reflect ``velocity`` off ``wall`` (shape (2, 2)) as ``ratinabox.utils.wall_bounce``."""
    parallel = wall[1] - wall[0]
    perpendicular = np.array([-parallel[1], parallel[0]])
    if np.dot(perpendicular, velocity) <= 0.0:
        perpendicular = -perpendicular
    if np.dot(parallel, velocity) <= 0.0:
        parallel = -parallel
    parallel = parallel / np.linalg.norm(parallel)
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    return parallel * np.dot(velocity, parallel) - perpendicular * np.dot(velocity, perpendicular)


def closest_points(point: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Return the closest point on each wall segment (shape (N, 2, 2)) to ``point``."""
    start = walls[:, 0]
    direction = walls[:, 1] - start
    length_sq = np.einsum("ij,ij->i", direction, direction)
    fraction = np.einsum("ij,ij->i", point - start, direction) / np.where(length_sq > 0.0, length_sq, 1.0)
    fraction = np.clip(np.where(length_sq > 0.0, fraction, 0.0), 0.0, 1.0)
    return start + fraction[:, None] * direction


@dataclass
class StepResult:
    """Outcome of one motion update."""

    net_velocity: np.ndarray
    collision_wall: Optional[np.ndarray]
    closest_wall_distance: float


class RatMotion:
    """Stateful RatInABox motion model for a disc-shaped robot of radius ``robot_radius``.

    ``collision_margin`` only enlarges the body used for collision detection, to absorb sensor
    latency and braking distance; wall repulsion still uses the bare radius as in simulation.
    """

    def __init__(self, params: MotionParams, robot_radius: float, heading: float, rng: np.random.Generator, collision_margin: float = 0.0) -> None:
        self.params = params
        self.robot_radius = robot_radius
        self.collision_margin = collision_margin
        self.rng = rng
        self.rotational_velocity = 0.0
        self.velocity = params.speed_mean / 1e3 * np.array([cos(heading), sin(heading)])

    def step(self, walls: Sequence[Segment] | np.ndarray, dt: float) -> StepResult:
        """Advance the internal velocity by ``dt`` and return the displacement velocity to execute.

        ``walls`` are obstacle-surface segments relative to the robot centre. Distances to them
        are reduced by the robot radius so that the model sees the same shrunken box as the
        simulation, whose arena is the room minus the robot radius on every side.
        """
        p = self.params
        self.rotational_velocity += ornstein_uhlenbeck(dt, self.rotational_velocity, 0.0, p.rotational_velocity_std, p.rotational_velocity_coherence_time, self.rng)
        self.velocity = rotate(self.velocity, self.rotational_velocity * dt)

        speed = float(np.linalg.norm(self.velocity))
        if speed == 0.0:
            self.velocity, speed = np.array([1e-8, 0.0]), 1e-8
        z = rayleigh_to_normal(speed, p.speed_mean)
        z += ornstein_uhlenbeck(dt, z, 0.0, 1.0, p.speed_coherence_time, self.rng)
        self.velocity = normal_to_rayleigh(z, p.speed_mean) / speed * self.velocity

        wall_array = np.asarray(walls, dtype=float).reshape(-1, 2, 2)
        origin = np.zeros(2)
        wall_speed = np.zeros(2)
        closest = float("inf")
        if len(wall_array) > 0:
            vectors = origin - closest_points(origin, wall_array)
            centre_distances = np.linalg.norm(vectors, axis=1)
            valid = centre_distances > 0.0
            vectors, centre_distances = vectors[valid], centre_distances[valid]
            if len(centre_distances) > 0:
                normals = vectors / centre_distances[:, None]
                x = np.maximum(centre_distances - self.robot_radius, 0.0)
                closest = float(x.min())
                d = p.wall_repel_distance
                v = p.wall_repel_strength * p.speed_mean
                if p.wall_repel_strength > 0.0:
                    near = x <= d
                    accelerations = np.where(near, v ** 2 / d ** 2 * (d - x), 0.0)
                    self.velocity = self.velocity + 3.0 * (1.0 - p.thigmotaxis) ** 2 * (accelerations[:, None] * normals).sum(axis=0) * dt
                    speeds = np.where(near, v * (1.0 - np.sqrt(np.clip(1.0 - (d - x) ** 2 / d ** 2, 0.0, None))), 0.0)
                    wall_speed = (speeds[:, None] * normals).sum(axis=0)

        net_velocity = self.velocity + 6.0 * p.thigmotaxis ** 2 * wall_speed
        return StepResult(net_velocity, self.colliding_wall(wall_array, net_velocity * dt), closest)

    def colliding_wall(self, walls: Sequence[Segment] | np.ndarray, displacement: np.ndarray) -> Optional[np.ndarray]:
        """Return the wall the robot body would penetrate further by moving ``displacement``."""
        walls = np.asarray(walls, dtype=float).reshape(-1, 2, 2)
        if len(walls) == 0:
            return None
        origin = np.zeros(2)
        before = np.linalg.norm(closest_points(origin, walls), axis=1)
        after = np.linalg.norm(displacement - closest_points(displacement, walls), axis=1)
        hits = np.flatnonzero((after < self.robot_radius + self.collision_margin) & (after < before))
        if len(hits) == 0:
            return None
        return walls[hits[np.argmin(after[hits])]]

    def bounce(self, wall: np.ndarray) -> None:
        """Mirror the velocity off ``wall`` and slow down to half the mean speed, as RatInABox.

        A zero-length wall (an isolated obstacle point relative to the robot at the origin) is
        treated as a wall tangent to the robot at that point.
        """
        wall = np.asarray(wall, dtype=float)
        if np.allclose(wall[0], wall[1]):
            wall = np.array([wall[0], wall[0] + np.array([-wall[0][1], wall[0][0]])])
        reflected = wall_bounce(self.velocity, wall)
        self.velocity = 0.5 * self.params.speed_mean / np.linalg.norm(reflected) * reflected
