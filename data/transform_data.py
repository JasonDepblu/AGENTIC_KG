"""
Transform wide-format attention data to long-format for knowledge graph import.

The original data_process.csv has:
- Row 0: header with brand-powertrain combinations as column names
- Column 0: attribute names (e.g., "性价比", "外观风格")
- Column 1: "关注度均值" (average attention score)
- Columns 2+: attention scores for each brand-powertrain

This script transforms it into:
1. attributes.csv: unique attributes with their average attention scores
2. brands.csv: unique brand-powertrain combinations
3. attention_scores.csv: relationships between attributes and brands with scores
"""

import pandas as pd
from pathlib import Path


def transform_attention_data(input_file: str, output_dir: str):
    """
    Transform wide-format attention data to long-format.

    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Read the CSV
    df = pd.read_csv(input_file, index_col=0)

    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns[:5])}... ({len(df.columns)} total)")
    print(f"Index (attributes): {list(df.index[:5])}... ({len(df.index)} total)")

    # Extract average attention column
    avg_col = "关注度均值"

    # Create attributes.csv
    attributes_df = pd.DataFrame({
        "attribute_name": df.index,
        "average_attention": df[avg_col].values
    })
    attributes_file = output_path / "attributes.csv"
    attributes_df.to_csv(attributes_file, index=False)
    print(f"\nCreated {attributes_file} with {len(attributes_df)} attributes")

    # Extract brand-powertrain columns (exclude the average column)
    brand_cols = [col for col in df.columns if col != avg_col]

    # Parse brand and powertrain from column names
    brands_data = []
    for col in brand_cols:
        parts = col.split(" ", 1)
        if len(parts) == 2:
            brand, powertrain = parts
        else:
            brand = parts[0]
            powertrain = "unknown"
        brands_data.append({
            "brand_powertrain": col,
            "brand": brand,
            "powertrain": powertrain
        })

    brands_df = pd.DataFrame(brands_data)
    brands_file = output_path / "brands.csv"
    brands_df.to_csv(brands_file, index=False)
    print(f"Created {brands_file} with {len(brands_df)} brand-powertrain combinations")

    # Create attention_scores.csv (long format)
    scores_data = []
    for attr in df.index:
        for col in brand_cols:
            score = df.loc[attr, col]
            if pd.notna(score):
                scores_data.append({
                    "attribute_name": attr,
                    "brand_powertrain": col,
                    "attention_score": score
                })

    scores_df = pd.DataFrame(scores_data)
    scores_file = output_path / "attention_scores.csv"
    scores_df.to_csv(scores_file, index=False)
    print(f"Created {scores_file} with {len(scores_df)} attention score records")

    # Print sample data
    print("\n--- Sample Data ---")
    print("\nattributes.csv (first 5 rows):")
    print(attributes_df.head().to_string(index=False))

    print("\nbrands.csv (first 5 rows):")
    print(brands_df.head().to_string(index=False))

    print("\nattention_scores.csv (first 5 rows):")
    print(scores_df.head().to_string(index=False))

    return {
        "attributes": attributes_file,
        "brands": brands_file,
        "attention_scores": scores_file
    }


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "data_process.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    # Get the script's directory
    script_dir = Path(__file__).parent
    input_path = script_dir / input_file
    output_path = script_dir

    transform_attention_data(str(input_path), str(output_path))
