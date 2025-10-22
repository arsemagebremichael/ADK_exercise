
from google.adk.agents import Agent
from movie_recommender.rag_tool import recommend_movies


def movie_recommender_tool(query: str) -> str:
    """
    Recommends movies based on a natural language description.
    Args:
        query: A natural language description of the kind of movie the user wants.
    Returns:
        A personalized movie recommendation with explanation.
    """
    return recommend_movies(query)


root_agent = Agent(
    name="LoveSage",
    model="gemini-1.5-flash",
    description="A smart movie recommendation expert using RAG.",
    instruction=(
        "You are a friendly and knowledgeable movie buff. "
        "Use the movie_recommender_tool to answer user queries with personalized, "
        "well-explained recommendations. Be warm and engaging."
    ),
    tools=[movie_recommender_tool],
)