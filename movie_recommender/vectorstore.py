
# For our chromadb
import pandas as pd
import re
import spacy
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from pathlib import Path

# -------------------------------
# 1. Load and Parse Data
# -------------------------------
print("Loading and preprocessing data...")

# In vectorstore.py
df = pd.read_csv("cleaned_imdb_top_1000.csv") 

# Helper: Parse Cast field
def parse_cast_field(cast_str):
    if pd.isna(cast_str) or not isinstance(cast_str, str):
        return {"director": "", "stars": []}
    director_match = re.search(r"Director:\s*([^|]+)", cast_str)
    stars_match = re.search(r"Stars:\s*(.+)", cast_str)
    director = director_match.group(1).strip() if director_match else ""
    stars = [s.strip() for s in stars_match.group(1).split(",")] if stars_match else []
    return {"director": director, "stars": stars}

# Helper: Parse Info field (Votes and Gross)
def parse_info_field(info_str):
    if pd.isna(info_str) or not isinstance(info_str, str):
        return {"votes": 0, "gross": 0.0}
    votes_match = re.search(r"Votes:\s*([\d,]+)", info_str)
    gross_match = re.search(r"Gross:\s*\$(\d+\.?\d*)M", info_str)
    votes = int(votes_match.group(1).replace(",", "")) if votes_match else 0
    gross = float(gross_match.group(1)) if gross_match else 0.0  # in millions
    return {"votes": votes, "gross": gross}

# Apply parsing
df["parsed_cast"] = df["Cast"].apply(parse_cast_field)
df["Director"] = df["parsed_cast"].apply(lambda x: x["director"])
df["Stars"] = df["parsed_cast"].apply(lambda x: x["stars"])

df["parsed_info"] = df["Info"].apply(parse_info_field)
df["Votes"] = df["parsed_info"].apply(lambda x: x["votes"])
df["Gross"] = df["parsed_info"].apply(lambda x: x["gross"])

# -------------------------------
# 2. Clean Text with spaCy
# -------------------------------
print("Cleaning text for embeddings...")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "SpaCy model not found. Install it with:\n"
        "python -m spacy download en_core_web_sm"
    )

def clean_text(text: str) -> str:
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

# -------------------------------
# 3. Create MetaText for Embedding
# -------------------------------
print("Creating MetaText for semantic search...")

df["MetaText"] = df.apply(lambda row: (
    f"Title: {row['Title']}\n"
    f"Director: {row['Director']}\n"
    f"Genre: {row['Genre']}\n"
    f"Stars: {', '.join(row['Stars'])}\n"
    f"Plot: {row['Description']}\n"
    f"Rating: {row['Rate']}\n"
    f"Votes: {row['Votes']:,} | Gross: ${row['Gross']}M"
), axis=1)

# Clean MetaText
df["cleaned_metatext"] = df["MetaText"].apply(clean_text)

# -------------------------------
# 4. Generate Embeddings
# -------------------------------
print("Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["cleaned_metatext"].tolist(), show_progress_bar=True)

# -------------------------------
# 5. Store in ChromaDB (Persistent)
# -------------------------------
print("Storing in ChromaDB...")

# Use path relative to this file
db_path = Path(__file__).parent / "chroma_db"
client = PersistentClient(path=str(db_path))

# Create or get collection
collection = client.get_or_create_collection(name="movies")

# Add data
collection.add(
    embeddings=embeddings.tolist(),
    documents=df["MetaText"].tolist(),  # Original readable text
    metadatas=df[[
        "Title", "Genre", "Director", "Rate", "Votes", "Gross", "Duration"
    ]].to_dict(orient="records"),
    ids=[f"movie_{i}" for i in range(len(df))]
)

print("✅ ChromaDB: Movie data stored with metadata.")