import pandas as pd
import numpy as np
import re
import unicodedata
import spacy

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

    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    
    doc = nlp(text)

    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
    ]
    return " ".join(tokens)

def clean_movie_data(csv_path="IMDB top 1000.csv", output_path="cleaned_imdb_top_1000.csv"):
    df = pd.read_csv(csv_path, on_bad_lines='skip')

    df = df.replace(r'^\s*$', np.nan, regex=True)

    original_rows, original_cols = df.shape

    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > 0.5].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    required = ['Title', 'Description']
    missing_required = df[required].isnull().any(axis=1)
    df = df[~missing_required]
    dropped_rows = missing_required.sum()

    text_columns = ['Title', 'Description', 'Genre', 'Director']
    for col in text_columns:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].apply(lambda x: clean_text_lemmatize(str(x)) if pd.notna(x) else "")

    for col in text_columns:
        if f"{col}_clean" in df.columns:
            df[col] = df[f"{col}_clean"]
            df = df.drop(columns=[f"{col}_clean"])

    def extract_duration(duration):
        if pd.isna(duration):
            return np.nan
        match = re.search(r'(\d+)', str(duration))
        return int(match.group(1)) if match else np.nan

    if 'Duration' in df.columns:
        df['Duration'] = df['Duration'].apply(extract_duration)

    if 'Rate' in df.columns:
        df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
        df['Rate'] = df['Rate'].clip(0, 10)

    threshold = 0.5 * len(df)
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].isnull().sum() < threshold:
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

    final_rows, final_cols = df.shape

    df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    cleaned_df = clean_movie_data()
