import json
import anthropic
import asyncio
import aiohttp

# ---------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------
async def wiki_search_with_claude(user_input: str, include_full_content: bool = True) -> str:
    """
    Asynchronously:
    1. Generate search queries with Claude (synchronously).
    2. Collect information from up to 3 Wikipedia articles per query via aiohttp.
    3. Return the combined information as a string.
    
    Args:
        user_input: The topic to generate search queries for and retrieve info on.
        include_full_content: Whether to include the full article text or just intro.
        
    Returns:
        A single string containing the retrieved Wikipedia information.
    """
    # 1. Generate concept-focused search queries (synchronously).
    search_queries = generate_search_queries(user_input)

    # 2. Asynchronously search Wikipedia for each query and collect results.
    wiki_content = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_wikipedia_content(session, query, include_full_content)
            for query in search_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    # Collect results or errors
    for query, content in zip(search_queries, results):
        if isinstance(content, Exception):
            wiki_content.append(f"--- Error retrieving content for '{query}' ---\n{str(content)}")
        else:
            wiki_content.append(f"--- Content for '{query}' ---\n{content}")
    
    # 3. Combine all content into a single string
    if not wiki_content:
        return "No Wikipedia content found for the given input."
    
    return "\n\n".join(wiki_content)

# ---------------------------------------------------------------------
# CLAUDE QUERY GENERATION (SYNCHRONOUS)
# ---------------------------------------------------------------------
def generate_search_queries(topic: str):
    """
    Call Anthropic (synchronously) to get a list of concept-focused
    Wikipedia search queries for the given topic.
    
    Caching is removed to simplify and ensure fresh queries each time.
    """
    client = anthropic.Anthropic()
    
    # Prepare instructions and tool usage, if needed
    system_prompt = """
    You are an expert at generating effective search queries for educational content on Wikipedia.
    Based on the user's topic, generate 4-6 specific search queries that would yield
    comprehensive information about the core concepts, principles, and applications of the topic.
    
    Your queries should focus on:
    1. The main concept or algorithm directly
    2. Key theoretical principles and mechanisms
    3. Common variants or alternative approaches
    4. Practical applications and implementations
    
    Prioritize technical understanding of how the concept works over historical information or
    key contributors. The goal is to gather information suitable for creating educational
    flashcards and quiz questions that test comprehension of the topic.
    
    Keep queries concise but specific, and ensure they use terminology that would match
    Wikipedia article sections about the mechanism, process, or implementation.
    """

    tools = [
        {
            "name": "generate_wiki_queries",
            "description": "Generate relevant search queries for Wikipedia based on a topic",
            "input_schema": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "List of search queries for Wikipedia",
                        "items": {
                            "type": "string", 
                            "description": "A specific search query"
                        }
                    }
                },
                "required": ["queries"]
            }
        }
    ]

    # Create system + user prompts
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        temperature=0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": f"Generate Wikipedia search queries for the educational topic: {topic}"}
        ],
        tools=tools,
        tool_choice={"type": "tool", "name": "generate_wiki_queries"}
    )
    
    # Attempt to parse the structured output for queries
    queries = []
    if isinstance(response.content, list):
        # In some Anthropics Python clients, response.content might be list of messages
        for content in response.content:
            if getattr(content, "type", "") == "tool_use" and content.name == "generate_wiki_queries":
                queries = content.input.get("queries", [])
    else:
        # Fallback to just returning the topic as a single query
        queries = [topic]
    
    # If Anthropic didn't produce anything, fallback
    if not queries:
        queries = [topic]
    
    return queries

# ---------------------------------------------------------------------
# WIKIPEDIA RETRIEVAL (ASYNC + AIOHTTP)
# ---------------------------------------------------------------------
async def fetch_wikipedia_content(session: aiohttp.ClientSession, query: str, full_content: bool) -> str:
    """
    1. Search Wikipedia for up to 3 articles matching 'query'.
    2. For each article, fetch its content (intro or full).
    3. Return the combined content as a string.
    """
    # 1. Search for up to 3 matching articles
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
        "srlimit": 3  # Hard-limit to 3 articles
    }
    
    try:
        async with session.get(search_url, params=search_params, timeout=10) as resp:
            search_data = await resp.json()
    except Exception as e:
        return f"Error searching Wikipedia for '{query}': {str(e)}"
    
    search_results = search_data.get("query", {}).get("search", [])
    if not search_results:
        return f"No Wikipedia articles found for query: {query}"

    # 2. Fetch each article in parallel
    tasks = []
    for result in search_results:
        title = result["title"]
        tasks.append(fetch_article_content(session, title, full_content))
    
    article_texts = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 3. Combine articles
    combined_content = []
    for result, content in zip(search_results, article_texts):
        title = result["title"]
        if isinstance(content, Exception):
            combined_content.append(f"### {title} ###\nError: {str(content)}")
        else:
            combined_content.append(f"### {title} ###\n{content}")
    
    return "\n\n".join(combined_content)


async def fetch_article_content(session: aiohttp.ClientSession, title: str, full_content: bool = True) -> str:
    """
    Fetch either the intro or the full text of a Wikipedia article.
    """
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
    }
    
    if not full_content:
        # Only get the intro
        params["exintro"] = 1
    
    try:
        async with session.get(api_url, params=params, timeout=10) as resp:
            data = await resp.json()
    except Exception as e:
        return f"Error fetching content for '{title}': {str(e)}"
    
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return f"No content found for '{title}'"
    
    page_id = next(iter(pages))
    content = pages[page_id].get("extract", f"No content found for '{title}'")
    
    # Truncate lengthy articles
    if full_content and len(content) > 8000:
        content = content[:8000] + "... [content truncated]"
    
    return content

# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # You can run this script directly to test
    topic = "breadth first search"
    final_content = asyncio.run(wiki_search_with_claude(topic, include_full_content=True))
    print(final_content)
