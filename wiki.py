
import requests
import json
import anthropic
import concurrent.futures
import os
from typing import List, Dict, Any, Tuple

def wiki_search_with_claude(user_input: str, max_results_per_query: int = 3, include_full_content: bool = True) -> str:
    """
    Function that:
    1. Takes user input string
    2. Uses Claude Haiku to generate search queries focused on educational concepts
    3. Collects information from Wikipedia API searches
    4. Returns the combined information as a string, optimized for flashcard/quiz creation
    
    Args:
        user_input: User input topic string (e.g., "breadth first search")
        max_results_per_query: Maximum number of results to retrieve per query
        include_full_content: Whether to include full content (True) or just intro (False)
        
    Returns:
        A single string containing all Wikipedia information
    """
    # Step 1: Use Claude Haiku to generate concept-focused search queries
    search_queries = generate_search_queries(user_input)
    
    # Step 2: Search Wikipedia for each query in parallel
    wiki_content = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit search tasks for each query
        future_to_query = {
            executor.submit(search_wikipedia_multiple, query, max_results_per_query, include_full_content): query 
            for query in search_queries
        }
        
        # Process completed tasks as they finish
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                results = future.result()
                if results:
                    wiki_content.append(f"--- Content for '{query}' ---\n{results}")
            except Exception as e:
                wiki_content.append(f"--- Error retrieving content for '{query}' ---\n{str(e)}")
    
    # Step 3: Combine all content into a single string
    if not wiki_content:
        return "No Wikipedia content found for the given input."
    
    return "\n\n".join(wiki_content)

def generate_search_queries(topic: str) -> List[str]:
    """
    Use Claude Haiku to generate education-focused search queries for Wikipedia
    based on user input. Optimized for retrieving concept-focused content suitable
    for flashcards and quizzes.
    
    Args:
        topic: User input topic string
    
    Returns:
        List of search queries
    """
    # Simple caching mechanism - check if we've already processed this topic
    cache_file = f"query_cache_{hash(topic)}.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass  # If there's any error reading cache, continue with normal processing
    
    client = anthropic.Anthropic(
        # You'll need to set your API key here or use environment variables
        # api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Define the tool for structured output
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
    
    # Improved prompt focused on educational concepts for flashcards/quizzes
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
    
    # Call Claude Haiku with tool use for structured output
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
    
    # Extract the structured output
    queries = []
    for content in response.content:
        if content.type == "tool_use" and content.name == "generate_wiki_queries":
            queries = content.input.get("queries", [])
            
            # Save to cache for future use
            try:
                with open(cache_file, 'w') as f:
                    json.dump(queries, f)
            except:
                pass  # If we can't write to cache, just continue
                
            return queries
    
    # Fallback in case tool use failed
    fallback_queries = [topic]
    return fallback_queries

def search_wikipedia_multiple(query: str, max_results: int = 3, full_content: bool = True) -> str:
    """
    Search Wikipedia for a specific query and return content from multiple results.
    Focuses on retrieving educational content suitable for flashcards and quizzes.
    
    Args:
        query: Search query for Wikipedia
        max_results: Maximum number of results to return (default: 3)
        full_content: Whether to fetch full article content (True) or just intro (False)
        
    Returns:
        Combined content from multiple Wikipedia articles for the given query
    """
    # First, search for articles related to the query
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
        "srlimit": max_results  # Get multiple results
    }
    
    try:
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_data = search_response.json()
        
        # If no search results found
        if not search_data.get("query", {}).get("search", []):
            return f"No Wikipedia articles found for query: {query}"
        
        # Process multiple results in parallel
        search_results = search_data["query"]["search"]
        articles = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each article
            future_to_article = {
                executor.submit(
                    fetch_article_content, 
                    result["title"], 
                    full_content
                ): result["title"] 
                for result in search_results[:max_results]
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_article):
                title = future_to_article[future]
                try:
                    content = future.result()
                    if content:
                        articles.append(f"### {title} ###\n{content}")
                except Exception as e:
                    articles.append(f"### {title} ###\nError retrieving content: {str(e)}")
        
        # Combine all article contents
        return "\n\n".join(articles)
        
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

def fetch_article_content(title: str, full_content: bool = True) -> str:
    """
    Fetch content for a specific Wikipedia article, focusing on sections
    most relevant to educational understanding.
    
    Args:
        title: Title of the Wikipedia article
        full_content: Whether to fetch full article content (True) or just intro (False)
        
    Returns:
        Content of the Wikipedia article
    """
    api_url = "https://en.wikipedia.org/w/api.php"
    
    # Set parameters based on whether we want full content or just intro
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1  # Get plain text, not HTML
    }
    
    if not full_content:
        params["exintro"] = 1  # Only get the introduction section
    
    try:
        response = requests.get(api_url, params=params, timeout=10)
        data = response.json()
        
        # Extract the page content
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return f"Could not retrieve content for: {title}"
        
        # Get the content of the article
        page_id = next(iter(pages))
        content = pages[page_id].get("extract", f"No content found for: {title}")
        
        # For full content, filter out sections that aren't relevant for educational purposes
        if full_content:
            # This is a very basic approach - a more sophisticated version would
            # use the Wikipedia API to get the section structure and selectively include sections
            
            # Limit content length to avoid overwhelming results
            if len(content) > 8000:  # Increased from 5000 to include more educational content
                content = content[:8000] + "... [content truncated]"
            
        return content
    
    except Exception as e:
        return f"Error fetching article content: {str(e)}"

def fetch_sections(title: str) -> List[Dict]:
    """
    Fetch the section structure of a Wikipedia article.
    This can be used to more intelligently filter content for educational purposes.
    
    Args:
        title: Title of the Wikipedia article
        
    Returns:
        List of sections with their titles and indices
    """
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "sections"
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=10)
        data = response.json()
        
        if "parse" in data and "sections" in data["parse"]:
            return data["parse"]["sections"]
        return []
    
    except Exception:
        return []