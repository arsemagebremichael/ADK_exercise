# 🎬 LoveSage: AI Movie Recommender

> "Recommend me emotionally animated movies about food."  
> → *"I highly recommend Ratatouille, a heartwarming story about a rat who dreams of becoming a chef..."*

LoveSage is a **RAG-powered movie recommendation system** that understands natural language and delivers **smart, explained suggestions**.

Built with **Google ADK**, **ChromaDB**, and **Gemini**, it combines semantic search with AI reasoning to act like a real film expert.

---

## Features

- **Natural language queries**  
  _"Best psychological thrillers with twist endings"_
- **Semantic search**  
  Finds movies by meaning, not just genre
- **AI-generated explanations**  
  Powered by Gemini
- **Interactive UI**  
  Run in terminal or browser with ADK Dev UI

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|--------|
| `google-adk` | Agent framework & tool calling |
| `Gemini 1.5 Flash` | LLM for natural responses |
| `ChromaDB` | Vector database for semantic search |
| `Sentence Transformers` | Embedding model (`all-MiniLM-L6-v2`) |
| `spaCy` | Text cleaning & lemmatization |
| `pandas` | Data parsing & preprocessing |

---

## Try It

```bash
# Run in terminal
adk run movie_recommender

# Or use the web UI
adk web movie_recommender
