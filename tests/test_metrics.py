from handoff import metrics


def test_confusion_counts_each_cell_at_threshold():
    # Arrange: p >= 0.5 goes strong. tp=2, tn=1, fp=1, fn=1
    p = [0.9, 0.6, 0.7, 0.2, 0.4]
    y = [1, 1, 0, 0, 1]

    # Act
    out = metrics.confusion(p, y, 0.5)

    # Assert
    assert out == {"accuracy": 0.6, "strong_recall": 0.667, "cheap_recall": 0.5,
                   "hard_sent_cheap": 1, "easy_sent_strong": 1, "n": 5}


def test_confusion_threshold_is_inclusive_for_strong():
    out = metrics.confusion([0.5], [1], 0.5)

    assert out["hard_sent_cheap"] == 0 and out["accuracy"] == 1.0


def test_confusion_on_empty_input_does_not_divide_by_zero():
    out = metrics.confusion([], [], 0.5)

    assert out["accuracy"] == 0.0 and out["n"] == 0


def test_gate_table_to_cheap_is_monotonic_non_decreasing():
    p = [0.01, 0.05, 0.11, 0.2, 0.33, 0.45, 0.6, 0.9]
    y = [0, 0, 1, 0, 1, 0, 1, 1]

    table = metrics.gate_table(p, y)

    shares = [row["to_cheap"] for row in table]
    leaks = [row["hard_leak"] for row in table]
    assert [row["cut"] for row in table] == metrics.GATE_CUTS
    assert shares == sorted(shares)
    assert leaks == sorted(leaks)


def test_gate_table_reports_share_and_leak_for_a_cut():
    p = [0.01, 0.03, 0.03, 0.9]
    y = [0, 1, 0, 1]

    row = next(r for r in metrics.gate_table(p, y) if r["cut"] == 0.04)

    assert row == {"cut": 0.04, "to_cheap": 0.75, "hard_leak": 0.5}


def test_pick_gate_returns_largest_cut_within_max_leak():
    # The hard task at 0.25 starts leaking at cut 0.26.
    p = [0.01, 0.05, 0.1, 0.25, 0.8]
    y = [0, 0, 0, 1, 1]

    gate = metrics.pick_gate(p, y, max_leak=0.0)

    assert gate == 0.24


def test_pick_gate_allows_more_leak_when_budget_is_larger():
    p = [0.01, 0.05, 0.1, 0.25, 0.8]
    y = [0, 0, 0, 1, 1]

    assert metrics.pick_gate(p, y, max_leak=0.5) == 0.5


def test_pick_gate_returns_zero_when_no_cut_is_safe():
    # A hard task below the smallest cut leaks at every cut.
    p = [0.0, 0.9]
    y = [1, 1]

    assert metrics.pick_gate(p, y, max_leak=0.05) == 0.0


def _rows(n_pos, n_neg):
    return [(f"pos{i}", 1) for i in range(n_pos)] + [(f"neg{i}", 0) for i in range(n_neg)]


def test_split3_is_stratified_by_label():
    rows = _rows(20, 30)

    train, val, test = metrics.split3(rows)

    assert sum(l for _, l in test) == 4 and len(test) == 10
    assert sum(l for _, l in val) == 4 and len(val) == 10
    assert sum(l for _, l in train) == 12 and len(train) == 30


def test_split3_parts_are_disjoint_and_cover_everything():
    rows = _rows(20, 30)

    train, val, test = metrics.split3(rows)

    a, b, c = set(train), set(val), set(test)
    assert not (a & b or a & c or b & c)
    assert a | b | c == set(rows)


def test_split3_is_deterministic_for_a_seed_and_varies_with_seed():
    rows = _rows(20, 30)

    assert metrics.split3(rows) == metrics.split3(list(rows))
    assert metrics.split3(rows, seed=1) != metrics.split3(rows, seed=2)
