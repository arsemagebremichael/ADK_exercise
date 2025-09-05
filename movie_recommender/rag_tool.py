
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from movie_recommender.utils import clean_text  # ✅ Absolute import

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Connect to ChromaDB
db_path = Path(__file__).parent / "chroma_db"
client = chromadb.PersistentClient(path=str(db_path))
collection = client.get_collection(name="movies")

# Prompt template
PROMPT_TEMPLATE = """
You are a passionate and insightful movie recommendation expert.

User Query: "{query}"

Here are the most relevant movies based on plot, genre, director, cast, and tone:

{context}

Respond in a warm, conversational way:
- Recommend 1–3 movies.
- Explain why each matches (e.g., similar themes, mood, director, actors).
- Mention standout elements: story, performances, emotional impact.
- Avoid bullet points. Be natural and engaging.
"""

def recommend_movies(query: str) -> str:
    cleaned_query = clean_text(query)
    query_embedding = embedding_model.encode([cleaned_query])[0].tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # Format context
    context_parts = []
    for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
        stars = ", ".join(meta.get("Stars", [])) if isinstance(meta.get("Stars"), list) else meta.get("Stars", "")
        context_parts.append(
            f"**{meta['Title']}** ({meta['Duration']} min)\n"
            f"Genre: {meta['Genre']}\n"
            f"Director: {meta['Director']}\n"
            f"Stars: {stars}\n"
            f"Plot: {doc.split('Plot: ')[1].split('Rating:')[0].strip()}\n"
            f"Rating: {meta['Rate']}/10"
        )

    context = "\n\n".join(context_parts)

    # Generate with Gemini
    final_prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    try:
        response = gemini_model.generate_content(final_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry, I couldn't generate a recommendation due to an error: {str(e)}"