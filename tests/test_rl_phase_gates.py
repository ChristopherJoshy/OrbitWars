from src.ow.rl.phase_gates import phase_gate_satisfied, ship_waste_loss


def _row(t, rank=1.0, wasted=0, sun=0, mode="2p", gate_eval=False, gate_opponent_version=None):
    row = {
        "wall_time": t,
        "player0_rank": rank,
        "wasted_ships": wasted,
        "sun_loss_ships": sun,
        "phase_mode": mode,
        "gate_eval": gate_eval,
    }
    if gate_opponent_version is not None:
        row["gate_opponent_version"] = gate_opponent_version
    return row


def test_ship_waste_loss_uses_non_capture_waste_without_double_counting_sun():
    assert ship_waste_loss({"wasted_ships": 15, "sun_loss_ships": 10}) == 15
    assert ship_waste_loss({"wasted_ships": 0, "sun_loss_ships": 12}) == 12


def test_phase_gate_requires_30_minutes_of_strict_wins_and_low_ship_waste():
    rows = [_row(0), _row(900), _row(1800)]

    result = phase_gate_satisfied(rows, phase_mode="2p", min_duration_seconds=1800, max_ship_waste=20)

    assert result.satisfied is True
    assert result.window_seconds == 1800
    assert result.episodes == 3


def test_phase_gate_rejects_ties_losses_high_waste_and_wrong_mode():
    assert phase_gate_satisfied([_row(0), _row(1800, rank=1.5)], phase_mode="2p").satisfied is False
    assert phase_gate_satisfied([_row(0), _row(1800, wasted=21)], phase_mode="2p").satisfied is False
    assert phase_gate_satisfied([_row(0, mode="4p"), _row(1800, mode="4p")], phase_mode="2p").satisfied is False


def test_phase_gate_accepts_user_required_ship_waste_boundary_of_20_or_less():
    result = phase_gate_satisfied([_row(0), _row(1800, wasted=20)], phase_mode="2p", max_ship_waste=20)

    assert result.satisfied is True


def test_phase_gate_missing_timestamps_cannot_satisfy_duration():
    result = phase_gate_satisfied([{"player0_rank": 1.0, "wasted_ships": 0, "phase_mode": "2p"}], phase_mode="2p")

    assert result.satisfied is False
    assert result.reason == "missing_wall_time"


def test_phase_gate_uses_deterministic_gate_eval_rows_when_present():
    rows = [
        _row(0, gate_eval=True),
        _row(600, rank=2.0, gate_eval=False),
        _row(900, gate_eval=True),
        _row(1200, rank=2.0, gate_eval=False),
        _row(1800, gate_eval=True),
    ]

    result = phase_gate_satisfied(rows, phase_mode="2p", min_duration_seconds=1800, max_ship_waste=20)

    assert result.satisfied is True
    assert result.window_seconds == 1800
    assert result.episodes == 3


def test_phase_gate_does_not_stitch_wins_across_gate_opponent_versions():
    rows = [
        _row(0, gate_eval=True, gate_opponent_version=1),
        _row(900, gate_eval=True, gate_opponent_version=1),
        _row(1800, gate_eval=True, gate_opponent_version=2),
        _row(2700, gate_eval=True, gate_opponent_version=2),
    ]

    result = phase_gate_satisfied(rows, phase_mode="2p", min_duration_seconds=1800, max_ship_waste=20)

    assert result.satisfied is False
    assert result.reason == "duration_too_short"
    assert result.window_seconds == 900
