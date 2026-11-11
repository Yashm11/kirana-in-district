"""Tests for district-name matching, A4 clustering and the A5 opportunity index.

The tables here are tiny made-up fixtures for testing the maths only; the pipeline itself
never uses invented data.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.a4_cluster_districts import run_clustering
from src.analytics.a5_whitespace import cluster_category_spend, opportunity_index
from src.utils.names import match_names, normalise_name, suggest_matches


def test_normalise_name() -> None:
    assert normalise_name("  Bengaluru  Urban ") == "bengaluru urban"
    assert normalise_name("Dadra & Nagar Haveli") == "dadra and nagar haveli"
    assert normalise_name("Pune District") == "pune"


def test_match_names_ignores_stray_spaces() -> None:
    matched, unmatched = match_names(["Farrukhabad"], ["Farrukhab ad"])
    assert matched == {"Farrukhabad": "Farrukhab ad"} and unmatched == []


def test_match_names_exact_override_and_unmatched() -> None:
    reference = ["Bangalore Urban", "Pune"]
    matched, unmatched = match_names(
        ["Pune District", "Bengaluru Urban", "Nowhere"],
        reference,
        overrides={"bengaluru urban": "Bangalore Urban"},
    )
    assert matched == {"Pune District": "Pune", "Bengaluru Urban": "Bangalore Urban"}
    assert unmatched == ["Nowhere"]


def test_suggestions_are_never_applied() -> None:
    table = suggest_matches(["Bangalor Urban"], ["Bangalore Urban", "Pune"])
    assert table.loc[0, "suggested_match"] == "Bangalore Urban"
    assert table.loc[0, "standard_name"] == ""  # a human fills this in


def test_clustering_finds_three_obvious_groups() -> None:
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0, 0], [10, 10, 10], [-10, 10, -10]])
    X = np.vstack([c + rng.normal(0, 0.5, (30, 3)) for c in centres])
    res = run_clustering(X, variance_target=0.85, k_min=2, k_max=6, seed=42)
    assert res.k == 3
    assert len(set(res.labels[:30])) == 1  # each blob gets one label


def test_opportunity_index() -> None:
    spend = pd.DataFrame(
        {
            "cluster": [0, 1],
            "category": ["tea", "tea"],
            "spend_per_capita": [100.0, 60.0],
            "population": [1000, 2000],
        }
    )
    out = opportunity_index(spend).set_index("cluster")
    assert out.loc[0, "opportunity_index"] == 0  # the benchmark cluster has no gap
    assert out.loc[1, "opportunity_index"] == pytest.approx((100 - 60) * 2000)
    assert out.loc[1, "relative_score"] == 100
    assert out.loc[1, "benchmark_cluster"] == 0


def test_cluster_spend_is_population_weighted() -> None:
    d = pd.DataFrame(
        {
            "cluster": [0, 0],
            "population_est": [1000, 3000],
            "spend_tea": [10.0, 30.0],
        }
    )
    out = cluster_category_spend(d, {"tea": "spend_tea"})
    assert out.loc[0, "spend_per_capita"] == pytest.approx(25.0)
    assert out.loc[0, "population"] == 4000


def test_sensitivity_and_profile_on_three_groups() -> None:
    from src.analytics.a4_cluster_districts import profile_clusters, sensitivity_check
    from src.utils.paths import load_config

    rng = np.random.default_rng(1)
    centres = np.array([[0, 0, 0], [10, 10, 10], [-10, 10, -10]])
    X = np.vstack([c + rng.normal(0, 0.5, (30, 3)) for c in centres])
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"]).assign(population_est=1000)
    cfg = load_config()
    th = cfg["thresholds"]
    base = run_clustering(
        X, th["pca_variance_target"], th["kmeans_k_min"], th["kmeans_k_max"], seed=42
    )
    sens = sensitivity_check(df, ["f1", "f2", "f3"], base, cfg, seed=42)
    expected = {f"k={base.k - 1}", f"k={base.k + 1}", "without f1", "without f2", "without f3"}
    assert set(sens["variant"]) == expected
    prof = profile_clusters(df, base.labels, ["f1", "f2", "f3"], "population_est")
    assert prof["n_districts"].sum() == 90 and prof["population"].sum() == 90_000


def test_impute_population_shares_state_leftover_then_falls_back() -> None:
    from src.analytics.a3_district_features import impute_population

    df = pd.DataFrame(
        {
            "state_code": [1, 1, 1, 2, 2, 2],
            "population": [100.0, 200.0, np.nan, 100.0, 100.0, np.nan],
            "state_persons": [500.0, 500.0, 500.0, 200.0, 200.0, 200.0],
        }
    )
    out = impute_population(df)
    assert out.loc[2, "population"] == pytest.approx(200.0)  # 500 - 300 left over
    assert out.loc[5, "population"] == pytest.approx(100.0)  # nothing left: median district
    assert out["population_imputed"].tolist() == [False, False, True, False, False, True]


def test_round_pack_size_snaps_to_common_sizes() -> None:
    from src.analytics.a5_whitespace import round_pack_size

    assert round_pack_size(98) == 100
    assert round_pack_size(180) == 200
    assert round_pack_size(1100) == 1000


def test_recommendations_use_cheaper_entry_for_lower_spend_clusters() -> None:
    from src.analytics.a5_whitespace import build_recommendations

    opp = opportunity_index(
        pd.DataFrame(
            {
                "cluster": [0, 1],
                "category": ["tea", "tea"],
                "spend_per_capita": [50.0, 100.0],  # cluster 0 spends 50% of the benchmark
                "population": [1000, 1000],
            }
        )
    )
    entry = pd.DataFrame(
        {"target_category": ["tea"] * 4, "entry_price_rs": [10.0, 20.0, 30.0, 40.0]}
    )
    gaps = pd.DataFrame(
        {
            "target_category": ["tea"],
            "pack_basis": ["g"],
            "from_pack": [50.0],
            "to_pack": [500.0],
            "is_gap": [True],
        }
    )
    ladders = pd.DataFrame(
        {
            "target_category": ["tea", "tea"],
            "pack_basis": ["g", "g"],
            "smallest_pack": [25.0, 25.0],
            "pack_premium_x": [1.5, 1.5],
        }
    )
    # observed packs: four near 25 g (Rs 10, 20, 30, 40) and four near 150 g (Rs 60-90)
    price_ladders = pd.DataFrame(
        {
            "target_category": ["tea"] * 8,
            "pack_basis": ["g"] * 8,
            "pack_total": [25.0, 25.0, 30.0, 20.0, 140.0, 150.0, 160.0, 150.0],
            "selling_price": [10.0, 20.0, 30.0, 40.0, 60.0, 70.0, 80.0, 90.0],
            "unit_price": [40.0, 80.0, 100.0, 200.0, 43.0, 47.0, 50.0, 60.0],
        }
    )
    recs = build_recommendations(opp, entry, gaps, ladders, price_ladders).set_index("cluster")
    assert recs.loc[0, "suggested_pack_size"] == "25 g"  # lower spend: entry pack
    assert recs.loc[1, "suggested_pack_size"] == "150 g"  # inside the 50 g -> 500 g gap
    assert recs.loc[0, "suggested_price_point"] == 18  # 25th percentile of Rs 10-40 (17.5)
    assert recs.loc[1, "suggested_price_point"] == 75  # median of Rs 60-90
    assert recs.loc[0, "priority_rank"] == 1
    assert recs.loc[0, "confidence"] == "low"  # only 2 ladders


def test_split_shared_population_and_dropping_copied_rows() -> None:
    from src.analytics.a3_district_features import drop_copied_rows, split_shared_population

    df = pd.DataFrame(
        {
            "district": ["Parent", "Child", "Alone"],
            "state_code": [9, 9, 9],
            "hces_district_code": [1.0, 1.0, 2.0],
            "population": [300.0, 300.0, 50.0],
            "urban_persons": [60.0, 60.0, 10.0],
        }
    )
    out = split_shared_population(df)
    assert out["population"].tolist() == [150.0, 150.0, 50.0]
    assert out["n_sharing"].tolist() == [2, 2, 1]
    assert (out["urban_persons"] / out["population"]).round(2).tolist() == [0.2, 0.2, 0.2]

    nfhs = pd.DataFrame(
        {"state": ["A", "B"], "district": ["X", "X"], "v": [1.0, 1.0], "w": [5.0, 5.0]}
    )
    assert drop_copied_rows(nfhs, ["v", "w"])["state"].tolist() == ["A"]


def test_relabel_by_feature_orders_clusters() -> None:
    from src.analytics.a4_cluster_districts import relabel_by_feature

    labels = np.array([2, 2, 0, 0, 1])
    values = pd.Series([90.0, 80.0, 10.0, 20.0, 50.0])
    assert relabel_by_feature(labels, values).tolist() == [2, 2, 0, 0, 1]
    assert relabel_by_feature(np.array([1, 1, 0]), pd.Series([5.0, 6.0, 50.0])).tolist() == [
        0, 0, 1,
    ]  # fmt: skip
