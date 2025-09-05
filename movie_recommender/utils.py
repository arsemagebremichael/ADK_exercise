import pandas as pd
import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "SpaCy model not found. Install it with:\n"
        "python -m spacy download en_core_web_sm"
    )

def parse_cast_field(cast_str: str):
    """
    Parse 'Director: X | Stars: Y, Z' into a dictionary.
    """
    if pd.isna(cast_str) or not isinstance(cast_str, str):
        return {"director": "", "stars": []}
    director_match = re.search(r"Director:\s*([^|]+)", cast_str)
    stars_match = re.search(r"Stars:\s*(.+)", cast_str)
    director = director_match.group(1).strip() if director_match else ""
    stars = [s.strip() for s in stars_match.group(1).split(",")] if stars_match else []
    return {"director": director, "stars": stars}

def parse_info_field(info_str: str):
    """
    Parse 'Votes: 2,295,987 | Gross: $28.34M' into a dictionary.
    """
    if pd.isna(info_str) or not isinstance(info_str, str):
        return {"votes": 0, "gross": 0.0}
    votes_match = re.search(r"Votes:\s*([\d,]+)", info_str)
    gross_match = re.search(r"Gross:\s*\$(\d+\.?\d*)M", info_str)
    votes = int(votes_match.group(1).replace(",", "")) if votes_match else 0
    gross = float(gross_match.group(1)) if gross_match else 0.0  # in millions
    return {"votes": votes, "gross": gross}

def clean_text(text: str) -> str:
    """
    Lemmatize and clean text using spaCy.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    doc = nlp(text.lower())
    lemmas = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
    ]
    return " ".join(lemmas)