from src.ow import chadman


def test_bandit_disabled_env_helper(monkeypatch):
    monkeypatch.delenv("OW_DISABLE_META_BANDIT", raising=False)
    assert chadman.meta_bandit_enabled() is True

    monkeypatch.setenv("OW_DISABLE_META_BANDIT", "1")
    assert chadman.meta_bandit_enabled() is False

    monkeypatch.setenv("OW_DISABLE_META_BANDIT", "TRUE")
    assert chadman.meta_bandit_enabled() is False

    monkeypatch.setenv("OW_DISABLE_META_BANDIT", "yes")
    assert chadman.meta_bandit_enabled() is False

    monkeypatch.setenv("OW_DISABLE_META_BANDIT", "On")
    assert chadman.meta_bandit_enabled() is False


def test_bandit_disabled_policy_is_neutral_even_with_stale_state(monkeypatch):
    monkeypatch.setenv("OW_DISABLE_META_BANDIT", "1")
    monkeypatch.setattr(
        chadman._META_BANDIT,
        "q",
        {arm: chadman.BANDIT_Q_CLIP for arm in chadman._META_ARMS},
    )

    policy = chadman._META_BANDIT.policy(object())

    assert policy["expansion_value"] == 1.0
    assert policy["pressure_value"] == 1.0
    assert policy["retention_value"] == 1.0
    assert policy["comet_value"] == 1.0
    assert policy["safety_value"] == 1.0
    assert policy["weakest_value"] == 1.0
    assert policy["leader_owner"] is None
    assert policy["leader_tax"] == 1.0
    assert policy["send_margin_bias"] == 0.0
    assert policy["packet_speed_bias"] == 0.0
    assert all(value == 0.0 for value in policy["q"].values())


def test_compute_soft_budget_tightens_when_overage_low():
    normal = chadman.compute_soft_budget(act_timeout=1.0, remaining_overage=60.0)
    tight = chadman.compute_soft_budget(act_timeout=1.0, remaining_overage=2.0)

    assert 0.55 <= normal <= 0.90
    assert tight < normal
    assert tight >= 0.05
