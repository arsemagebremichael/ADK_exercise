import pandas as pd
import numpy as np
import re
import unicodedata
import spacy

#Loading spaCy model for lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "SpaCy model not found. Install it with:\n"
        "python -m spacy download en_core_web_sm"
    )

def clean_text_lemmatize(text: str) -> str:
    """Clean and lemmatize text using spaCy."""
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Normalize Unicode and clean
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    
    # Parse with spaCy
    doc = nlp(text)

    # Lemmatize, remove stopwords/punctuation
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
    ]
    return " ".join(tokens)

def clean_movie_data(csv_path="IMDB top 1000.csv", output_path="cleaned_imdb_top_1000.csv"):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, on_bad_lines='skip')

    # Replace empty strings and whitespace with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    original_rows, original_cols = df.shape
    print(f"Original data: {original_rows} rows, {original_cols} columns")

    # -------------------------------
    # 1. Drop columns with >50% missing
    # -------------------------------
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > 0.5].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >50% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # -------------------------------
    # 2. Drop rows with missing Title or Description
    # -------------------------------
    required = ['Title', 'Description']
    missing_required = df[required].isnull().any(axis=1)
    df = df[~missing_required]
    dropped_rows = missing_required.sum()
    print(f"Dropped {dropped_rows} rows due to missing Title or Description")

    # 3. Normalize and Lemmatize Text Fields
    print("Normalizing and lemmatizing text fields...")
    text_columns = ['Title', 'Description', 'Genre', 'Director']
    for col in text_columns:
        if col in df.columns:
            print(f"  → Lemmatizing '{col}'...")
            df[f"{col}_clean"] = df[col].apply(lambda x: clean_text_lemmatize(str(x)) if pd.notna(x) else "")

    #Replace original columns with cleaned versions
    for col in text_columns:
        if f"{col}_clean" in df.columns:
            df[col] = df[f"{col}_clean"]
            df = df.drop(columns=[f"{col}_clean"])

    # -------------------------------
    # 4. Parse Duration: "142 min" → 142
    # -------------------------------
    def extract_duration(duration):
        if pd.isna(duration):
            return np.nan
        match = re.search(r'(\d+)', str(duration))
        return int(match.group(1)) if match else np.nan

    if 'Duration' in df.columns:
        df['Duration'] = df['Duration'].apply(extract_duration)


    # -------------------------------
    # 5. Validate Rating (0–10)
    # -------------------------------
    if 'Rate' in df.columns:
        df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
        df['Rate'] = df['Rate'].clip(0, 10)

    # -------------------------------
    # 6. Fill Remaining Missing Values (<50%)
    # -------------------------------
    threshold = 0.5 * len(df)  # 50% threshold
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].isnull().sum() < threshold:
            print(f"Filling '{col}' ({df[col].isnull().sum()} missing)")
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            elif col == 'Cast':
                df[col] = df[col].fillna('Director: Unknown | Stars: Unknown')
            elif col == 'Info':
                df[col] = df[col].fillna('Votes: 0 | Gross: $0M')
            else:
                df[col] = df[col].fillna('Unknown')
        else:
            print(f"Column '{col}' has >=50% missing — already dropped or kept as-is")

    # -------------------------------
    # 7. Final Info & Save
    # -------------------------------
    final_rows, final_cols = df.shape
    print(f"\nFinal cleaned data: {final_rows} rows ({((original_rows - final_rows)/original_rows)*100:.1f}% dropped)")
    print(f"Final columns: {final_cols} ({original_cols - final_cols} dropped)")
    print(f"Saving to {output_path}")
    df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    cleaned_df = clean_movie_data()
    print("\n✅ Data cleaning with lemmatization complete!")