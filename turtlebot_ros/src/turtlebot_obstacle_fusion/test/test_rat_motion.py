from math import pi, sqrt

import numpy as np
import pytest

from turtlebot_obstacle_fusion.rat_motion import (
    MotionParams,
    RatMotion,
    closest_points,
    normal_to_rayleigh,
    rayleigh_to_normal,
    scaled_params,
    wall_bounce,
)

S, K, R = 8.161, 7.248, 0.1705
DT = 0.02 * K


def test_scaled_params_follow_conversion_table() -> None:
    p = scaled_params("adult", S, K)
    assert p.speed_mean == pytest.approx(0.08021 * S / K)
    assert p.speed_coherence_time == pytest.approx(1.35 * K)
    assert p.rotational_velocity_std == pytest.approx(0.8334 * pi / K)
    assert p.rotational_velocity_coherence_time == pytest.approx(0.075 * K)
    assert p.wall_repel_distance == pytest.approx(0.8161)
    assert p.thigmotaxis == 0.4


def test_rayleigh_normal_round_trip() -> None:
    for speed in (0.01, 0.1, 0.3):
        assert normal_to_rayleigh(rayleigh_to_normal(speed, 0.1), 0.1) == pytest.approx(speed, rel=1e-6)


def test_free_motion_statistics_match_ou_targets() -> None:
    p = scaled_params("walk", S, K)
    model = RatMotion(p, R, 0.0, np.random.default_rng(0))
    speeds, omegas = [], []
    for _ in range(60000):
        model.step([], DT)
        speeds.append(np.linalg.norm(model.velocity))
        omegas.append(model.rotational_velocity)
    speeds, omegas = np.array(speeds[2000:]), np.array(omegas[2000:])
    assert np.mean(speeds) == pytest.approx(p.speed_mean * sqrt(pi / 2), rel=0.1)
    assert np.std(omegas) == pytest.approx(p.rotational_velocity_std, rel=0.1)


def test_wall_repulsion_pushes_away_and_keeps_wall_speed_out_of_velocity() -> None:
    p = MotionParams(0.1, 1e9, 1e-12, 1.0, 0.5, 0.8)
    model = RatMotion(p, R, pi / 2, np.random.default_rng(0))
    model.velocity = np.array([0.0, 0.1])
    wall = [((-5.0, 0.5 + R), (5.0, 0.5 + R))]
    result = model.step(wall, DT)
    assert model.velocity[1] < 0.1
    assert result.net_velocity[1] < model.velocity[1]
    assert result.closest_wall_distance == pytest.approx(0.5)


def test_walls_outside_repel_distance_are_ignored() -> None:
    p = MotionParams(0.1, 1e9, 1e-12, 1.0, 0.5, 0.8)
    model = RatMotion(p, R, 0.0, np.random.default_rng(0))
    model.velocity = np.array([0.1, 0.0])
    result = model.step([((-5.0, 2.0), (5.0, 2.0))], DT)
    assert np.allclose(result.net_velocity, model.velocity)


def test_collision_detected_only_when_moving_into_wall() -> None:
    p = MotionParams(0.1, 1e9, 1e-12, 1.0, 0.5, 0.8, wall_repel_strength=0.0)
    wall = [((-5.0, R + 0.005), (5.0, R + 0.005))]
    towards = RatMotion(p, R, pi / 2, np.random.default_rng(0))
    towards.velocity = np.array([0.0, 0.1])
    assert towards.step(wall, DT).collision_wall is not None
    along = RatMotion(p, R, 0.0, np.random.default_rng(0))
    along.velocity = np.array([0.1, 0.0])
    assert along.step(wall, DT).collision_wall is None


def test_bounce_mirrors_and_halves_speed() -> None:
    p = MotionParams(0.1, 1.0, 1.0, 1.0, 0.5, 0.8)
    model = RatMotion(p, R, 0.0, np.random.default_rng(0))
    model.velocity = np.array([0.3, 0.3])
    model.bounce(np.array([[-1.0, 1.0], [1.0, 1.0]]))
    assert np.allclose(model.velocity, 0.05 * np.array([1.0, -1.0]) / sqrt(2))


def test_wall_bounce_reflects_normal_component() -> None:
    assert np.allclose(wall_bounce(np.array([1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 0.0]])), [1.0, -2.0])


def test_closest_points_clip_to_segment_ends() -> None:
    walls = np.array([[[1.0, -1.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 1.0]], [[3.0, 3.0], [3.0, 3.0]]])
    assert np.allclose(closest_points(np.zeros(2), walls), [[1.0, 0.0], [1.0, 1.0], [3.0, 3.0]])


def test_bounce_off_isolated_point_reverses_head_on_motion() -> None:
    p = MotionParams(0.1, 1.0, 1.0, 1.0, 0.5, 0.8)
    model = RatMotion(p, R, 0.0, np.random.default_rng(0))
    model.velocity = np.array([0.2, 0.0])
    model.bounce(np.array([[0.15, 0.0], [0.15, 0.0]]))
    assert np.allclose(model.velocity, [-0.05, 0.0])


def test_module_colliding_wall_lets_robot_leave_a_touching_wall() -> None:
    from turtlebot_obstacle_fusion.rat_motion import colliding_wall
    wall = [((0.2, -1.0), (0.2, 1.0))]
    assert colliding_wall(wall, np.array([0.02, 0.0]), 0.22) is not None
    assert colliding_wall(wall, np.array([-0.02, 0.0]), 0.22) is None
    assert colliding_wall(wall, np.array([0.0, 0.02]), 0.22) is None


def _run(walls, merge, steps=40):
    from turtlebot_obstacle_fusion.rat_motion import RatMotion, scaled_params

    model = RatMotion(scaled_params("crawl", 8.161, 7.248), 0.1705, 0.0, np.random.default_rng(0), 0.05, merge)
    return [model.step(walls, 0.14496).net_velocity for _ in range(steps)]


def test_fragmented_wall_repels_like_a_single_wall_when_merged() -> None:
    single = np.array([[[0.30, -1.5], [0.30, 1.5]]])
    ys = np.linspace(-1.5, 1.5, 31)
    pieces = np.array([[[0.30, ys[i]], [0.30, ys[i + 1]]] for i in range(30)])
    assert np.allclose(_run(single, True), _run(pieces, True))
    assert np.linalg.norm(_run(pieces, False)[-1]) > 3.0 * np.linalg.norm(_run(single, False)[-1])


def test_box_corner_is_unchanged_by_merging() -> None:
    box = np.array([[[0.5, -2.0], [0.5, 2.0]], [[-2.0, 0.4], [2.0, 0.4]], [[-1.5, -2.0], [-1.5, 2.0]], [[-2.0, -1.8], [2.0, -1.8]]])
    assert np.allclose(_run(box, True), _run(box, False))


def test_convex_vertex_of_a_chain_counts_once() -> None:
    vertex = np.array([[[0.3, -0.5], [0.25, 0.0]], [[0.25, 0.0], [0.3, 0.5]]])
    single_point = np.array([[[0.25, 0.0], [0.25, 0.0]]])
    assert np.allclose(_run(vertex, True), _run(single_point, True))


def test_bounce_from_point_always_leads_away_from_the_contact() -> None:
    from turtlebot_obstacle_fusion.rat_motion import RatMotion, scaled_params

    params = scaled_params("walk", 8.161, 7.248)
    for velocity, point in [((0.1, 0.02), (0.17, 0.0)), ((0.1, 0.1), (0.12, 0.12)), ((-0.1, 0.0), (0.17, 0.0))]:
        model = RatMotion(params, 0.1705, 0.0, np.random.default_rng(0))
        model.velocity = np.array(velocity)
        model.bounce_from_point(np.array(point))
        assert np.dot(model.velocity, point) <= 0.0
        assert abs(np.linalg.norm(model.velocity) - 0.5 * params.speed_mean) < 1e-9

