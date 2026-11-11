"""Print the available pipeline stages: `uv run python -m src`."""

STAGES = [
    ("a0", "src.data.download_bigbasket", "Download data"),
    ("a1", "src.analytics.a1_clean_products", "Clean product data"),
    ("a2", "src.analytics.a2_price_pack", "Price-pack analysis"),
    ("a3", "src.analytics.a3_district_features", "District features"),
    ("a3b", "src.analytics.a3b_hces_unit_values", "HCES unit prices and quantities"),
    ("a3c", "src.analytics.a3c_kirana_density", "OSM kirana density per district"),
    ("a4", "src.analytics.a4_cluster_districts", "Cluster districts"),
    ("a5", "src.analytics.a5_whitespace", "Whitespace score and recommendations"),
]


def main() -> None:
    """List each stage with the command that runs it."""
    print("kirana-in-district pipeline stages:\n")
    for key, module, description in STAGES:
        print(f"  {key}  {description:<40} uv run python -m {module}")


if __name__ == "__main__":
    main()
