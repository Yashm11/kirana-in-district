"""A4: group districts into clusters with similar profiles.

Run: uv run python -m src.analytics.a4_cluster_districts     (or `make a4`)

Reads : data/processed/district_features.parquet (from A3) and the list of feature columns in
        config.yaml -> clustering.feature_columns
Writes: data/processed/district_clusters.parquet   district rows + `cluster` label
        data/processed/cluster_profiles.csv        one row per cluster (size, population, means)
        data/processed/cluster_sensitivity.csv     do clusters change if k or features change?

Method, in plain words
    1. StandardScaler: put every feature on the same scale (mean 0, spread 1), so a feature
       measured in thousands does not outweigh one measured in percent.
    2. PCA: squeeze correlated features into fewer "components" that still keep ~85% of the
       information (variance). Fewer, uncorrelated inputs make KMeans more stable.
    3. KMeans for k = 4..8; keep the k with the best silhouette score (how well separated and
       compact the clusters are; closer to 1 is better).
    4. Sensitivity: re-cluster with k-1 / k+1 and with each feature left out, and compare to
       the baseline with the Adjusted Rand Index (1 = identical grouping, 0 = random).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.log import get_logger
from src.utils.paths import get_dir, load_config

log = get_logger(__name__)


@dataclass
class ClusterResult:
    """Everything produced by one clustering run."""

    labels: np.ndarray  # cluster label for each input row
    k: int  # chosen number of clusters
    silhouette_by_k: pd.DataFrame  # columns: k, silhouette
    n_components: int  # PCA components kept
    variance_kept: float  # share of variance those components keep


def reduce_features(X: np.ndarray, variance_target: float, seed: int) -> tuple[np.ndarray, PCA]:
    """StandardScaler then PCA keeping `variance_target` of the variance."""
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=variance_target, svd_solver="full", random_state=seed)
    return pca.fit_transform(scaled), pca


def choose_k(scores: np.ndarray, k_min: int, k_max: int, seed: int) -> pd.DataFrame:
    """Silhouette score for each k in [k_min, k_max]."""
    rows = []
    for k in range(k_min, min(k_max, len(scores) - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(scores)
        rows.append({"k": k, "silhouette": silhouette_score(scores, labels)})
    return pd.DataFrame(rows)


def run_clustering(
    X: np.ndarray, variance_target: float, k_min: int, k_max: int, seed: int, k: int | None = None
) -> ClusterResult:
    """Scale, PCA, pick k by silhouette (unless `k` is given) and fit KMeans."""
    scores, pca = reduce_features(X, variance_target, seed)
    table = choose_k(scores, k_min, k_max, seed)
    best_k = k if k is not None else int(table.loc[table["silhouette"].idxmax(), "k"])
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit_predict(scores)
    return ClusterResult(
        labels, best_k, table, pca.n_components_, float(pca.explained_variance_ratio_.sum())
    )


def relabel_by_feature(labels: np.ndarray, values: pd.Series) -> np.ndarray:
    """Renumber clusters so 0 has the lowest mean of `values`, 1 the next lowest, and so on."""
    means = pd.Series(values.to_numpy()).groupby(labels).mean().sort_values()
    new_id = {old: new for new, old in enumerate(means.index)}
    return np.array([new_id[label] for label in labels])


def sensitivity_check(
    df: pd.DataFrame, feature_cols: list[str], base: ClusterResult, cfg: dict, seed: int
) -> pd.DataFrame:
    """Compare the baseline clusters with variants (k +/- 1, each feature dropped)."""
    th = cfg["thresholds"]
    args = (th["pca_variance_target"], th["kmeans_k_min"], th["kmeans_k_max"], seed)
    rows = []
    for k in (base.k - 1, base.k + 1):
        if k >= 2:
            alt = run_clustering(df[feature_cols].to_numpy(), *args, k=k)
            rows.append(
                {
                    "variant": f"k={k}",
                    "ari_vs_baseline": adjusted_rand_score(base.labels, alt.labels),
                }
            )
    if len(feature_cols) > 2:
        for col in feature_cols:
            rest = [c for c in feature_cols if c != col]
            alt = run_clustering(df[rest].to_numpy(), *args, k=base.k)
            rows.append(
                {
                    "variant": f"without {col}",
                    "ari_vs_baseline": adjusted_rand_score(base.labels, alt.labels),
                }
            )
    out = pd.DataFrame(rows)
    out["stable"] = out["ari_vs_baseline"] >= 0.7  # rule of thumb, not a statistical test
    return out


def profile_clusters(
    df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str], population_col: str | None
) -> pd.DataFrame:
    """One row per cluster: number of districts, population, and mean of each feature."""
    d = df.assign(cluster=labels)
    agg: dict[str, tuple[str, str]] = {"n_districts": (feature_cols[0], "size")}
    if population_col and population_col in d.columns:
        agg["population"] = (population_col, "sum")
    agg.update({c: (c, "mean") for c in feature_cols})
    return d.groupby("cluster").agg(**agg).reset_index()


def main() -> None:
    """Run A4 if A3's output and the feature list exist; otherwise say what is missing."""
    cfg = load_config()
    seed = cfg["random_seed"]
    feature_cols: list[str] = cfg["clustering"]["feature_columns"]
    src = get_dir("processed") / cfg["files"]["district_features"]
    if not src.exists() or not feature_cols:
        print("A4 SKIPPED:", end=" ")
        print(f"{src.name} not found (run A3)." if not src.exists() else "")
        if not feature_cols:
            print("config.yaml clustering.feature_columns is empty. Fill it after A3 runs.")
        return

    df = pd.read_parquet(src)
    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    th = cfg["thresholds"]
    res = run_clustering(
        df[feature_cols].to_numpy(),
        th["pca_variance_target"], th["kmeans_k_min"], th["kmeans_k_max"], seed,
    )  # fmt: skip
    order_by = cfg["clustering"].get("order_by")
    if order_by in df.columns:
        res.labels = relabel_by_feature(res.labels, df[order_by])
    sens = sensitivity_check(df, feature_cols, res, cfg, seed)
    profile = profile_clusters(df, res.labels, feature_cols, "population_est")

    out = get_dir("processed")
    df.assign(cluster=res.labels).to_parquet(out / cfg["files"]["district_clusters"], index=False)
    profile.to_csv(out / cfg["files"]["cluster_profiles"], index=False)
    sens.to_csv(out / cfg["files"]["cluster_sensitivity"], index=False)

    print("\n=== A4 summary ===")
    print(
        f"Districts in: {before:,}  used: {len(df):,}  "
        f"dropped (missing features): {before - len(df)}"
    )
    print(f"PCA: {res.n_components} components keep {res.variance_kept:.0%} of variance")
    print(res.silhouette_by_k.round(3).to_string(index=False))
    print(f"Chosen k = {res.k}\n")
    print(profile.round(2).to_string(index=False))
    print("\nSensitivity (ARI vs baseline; >= 0.7 treated as stable):")
    print(sens.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
