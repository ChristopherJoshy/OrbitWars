import json

from src.ow.rl.metrics import MetricsWriter, RollingStats
from src.ow.rl.rewards import mistake_penalty, shaped_reward


def test_shaped_reward_rewards_production_gain_and_penalizes_idle():
    prev = {"player": 0, "planets": [[0, 0, 0, 0, 1, 10, 2], [1, 1, 1, 0, 1, 10, 3]]}
    curr = {"player": 0, "planets": [[0, 0, 0, 0, 1, 10, 5], [1, 1, 1, 0, 1, 10, 3]]}
    assert shaped_reward(prev, curr, action_count=1, terminal_reward=0.0) > shaped_reward(prev, curr, action_count=0, terminal_reward=0.0)


def test_shaped_reward_penalizes_sun_waste_invalid_and_lost_planets():
    prev = {"player": 0, "planets": [[0, 0, 0, 0, 1, 10, 2], [1, 0, 1, 0, 1, 10, 2], [2, 1, 2, 0, 1, 10, 3]]}
    curr = {"player": 0, "planets": [[0, 0, 0, 0, 1, 10, 2], [1, 1, 1, 0, 1, 10, 2], [2, 1, 2, 0, 1, 10, 4]]}
    clean = shaped_reward(prev, curr, action_count=1, terminal_reward=0.0)
    punished = shaped_reward(
        prev,
        curr,
        action_count=1,
        terminal_reward=0.0,
        invalid_action=True,
        sun_loss_ships=10,
        wasted_ships=15,
        failed_attack_ships=8,
        oversend_ships=6,
        repeated_failure=True,
    )
    assert punished < clean - 0.5


def test_mistake_penalty_penalizes_waste_and_dumb_targeting_metrics():
    base = mistake_penalty({})
    punished = mistake_penalty(
        {
            "wasted_ships": 10,
            "wrong_target_ships": 10,
            "out_of_bounds_ships": 10,
            "invalid_actions": 1,
        }
    )
    assert punished < base - 0.5


def test_mistake_penalty_can_apply_extra_4p_pressure():
    components = {"wasted_ships": 10, "wrong_target_ships": 10}
    assert mistake_penalty(components, mode="4p") < mistake_penalty(components, mode="2p")


def test_mistake_penalty_strongly_penalizes_sun_and_ship_waste():
    penalty = mistake_penalty(
        {
            "sun_loss_ships": 10,
            "wasted_ships": 10,
            "wrong_target_ships": 10,
            "out_of_bounds_ships": 10,
            "invalid_actions": 1,
        },
        mode="2p",
    )

    assert penalty <= -2.0


def test_mistake_penalty_4p_pressure_is_substantially_stronger():
    components = {"sun_loss_ships": 10, "wasted_ships": 10, "wrong_target_ships": 10}

    assert mistake_penalty(components, mode="4p") <= 1.45 * mistake_penalty(components, mode="2p")


def test_metrics_writer_writes_jsonl(tmp_path):
    writer = MetricsWriter(tmp_path)
    writer.write({"update": 1, "wins": 2})
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows == [{"update": 1, "wins": 2}]


def test_metrics_writer_preserves_late_added_gate_fields_in_csv(tmp_path):
    writer = MetricsWriter(tmp_path)
    writer.write({"update": 1, "wins": 2})
    writer.write({"update": 2, "gate_action_margin_mean": 0.125})

    csv_text = (tmp_path / "metrics.csv").read_text()

    assert "gate_action_margin_mean" in csv_text.splitlines()[0]
    assert "0.125" in csv_text


def test_rolling_stats_recent_rate():
    stats = RollingStats(window=3)
    stats.add(True)
    stats.add(False)
    stats.add(True)
    assert stats.recent_win_rate == 2 / 3
