from aggregate_prediction_results import flatten


def test_flatten_expands_nested_resource_profile() -> None:
    profile = {
        "parameter_count": 342_771,
        "latency_batch1": {
            "median_ms": 3.1,
            "p90_ms": 3.4,
            "n": 100,
        },
    }

    assert flatten("", profile) == {
        "parameter_count": 342_771,
        "latency_batch1_median_ms": 3.1,
        "latency_batch1_p90_ms": 3.4,
        "latency_batch1_n": 100,
    }


def test_flatten_preserves_prefixes() -> None:
    assert flatten("test_", {"macro": {"f1": 0.8}}) == {
        "test_macro_f1": 0.8
    }
