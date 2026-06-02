from tools.run_chadman_feature_probes import summarize_probe_results


def test_summarize_probe_results_counts_passes_and_failures():
    rows = [
        {"name": "transfer", "passed": True},
        {"name": "bait", "passed": False, "reason": "no rescue"},
    ]

    summary = summarize_probe_results(rows)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
