# clean_data.py
import pandas as pd
import numpy as np
import os

def clean_movie_data(csv_path="IMDB top 1000.csv", output_path="cleaned_imdb_top_1000.csv"):
    """
    Cleans the movie dataset:
    - Drops rows with missing Title or Description
    - For other columns, if missing < 20%, fills with appropriate values
    - Saves cleaned data to output_path
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    original_rows = len(df)
    print(f"Original data: {original_rows} rows")

    # Replace empty strings, whitespace, and NaN with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # -------------------------------
    # 1. Drop rows where Title or Description is missing
    # -------------------------------
    required_columns = ['Title', 'Description']
    missing_in_required = df[required_columns].isnull().any(axis=1)
    df_clean = df[~missing_in_required].copy()

    dropped_rows = original_rows - len(df_clean)
    print(f"Dropped {dropped_rows} rows due to missing Title or Description")

    # -------------------------------
    # 2. Analyze missing values in other columns
    # -------------------------------
    missing_stats = df_clean.isnull().sum()
    total_rows = len(df_clean)
    print("\nMissing values after dropping empty Title/Description:")
    print(missing_stats[missing_stats > 0])

    # -------------------------------
    # 3. Fill columns with <20% missing
    # -------------------------------
    threshold = 0.2 * total_rows  # 20% of total rows

    for col in missing_stats.index:
        if missing_stats[col] == 0:
            continue

        if missing_stats[col] < threshold:
            print(f"Filling '{col}' ({missing_stats[col]} missing, {missing_stats[col]/total_rows:.1%})")

            if col in ['Genre', 'Director', 'Duration']:
                # Fill categorical with 'Unknown'
                df_clean[col] = df_clean[col].fillna('Unknown')
            elif col in ['Rate']:
                # Fill numeric with median
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"  → Filled with median: {median_val}")
            elif col == 'Cast':
                # Fill with 'Director: Unknown | Stars: Unknown'
                df_clean[col] = df_clean[col].fillna('Director: Unknown | Stars: Unknown')
            elif col == 'Info':
                # Fill with 'Votes: 0 | Gross: $0M'
                df_clean[col] = df_clean[col].fillna('Votes: 0 | Gross: $0M')
            else:
                # Generic fill
                df_clean[col] = df_clean[col].fillna('Unknown')
        else:
            print(f"Column '{col}' has {missing_stats[col]/total_rows:.1%} missing (>20%) — keeping as-is (no fill)")

    # -------------------------------
    # 4. Final info and save
    # -------------------------------
    print(f"\nFinal cleaned data: {len(df_clean)} rows ({((original_rows - len(df_clean))/original_rows)*100:.1f}% dropped)")
    print(f"Saving cleaned data to {output_path}")
    df_clean.to_csv(output_path, index=False)

    return df_clean

# Run the cleaning
if __name__ == "__main__":
    cleaned_df = clean_movie_data()
    print("\n✅ Data cleaning complete!")