from turtlebot_obstacle_fusion.geometry import adaptive_break_threshold, cluster_scan, rdp_segments


def test_adaptive_threshold_grows_with_range() -> None:
    assert adaptive_break_threshold(5.0, 0.02, 0.03, 2.25) > adaptive_break_threshold(1.0, 0.02, 0.03, 2.25)


def test_cluster_scan_splits_a_large_gap() -> None:
    clusters = cluster_scan([(1.0, 0.0), (1.01, 0.0), (2.0, 0.0), (2.01, 0.0)], [1.0, 1.01, 2.0, 2.01], 0.01, 0.01, 2.0, 2)
    assert clusters == [[(1.0, 0.0), (1.01, 0.0)], [(2.0, 0.0), (2.01, 0.0)]]


def test_rdp_keeps_a_corner() -> None:
    assert rdp_segments([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], 0.1) == [((0.0, 0.0), (1.0, 1.0)), ((1.0, 1.0), (2.0, 0.0))]
