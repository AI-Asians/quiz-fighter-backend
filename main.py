from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from wiki import wiki_search_with_claude
from generate_questions import generate_quiz_questions
import asyncio

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def hello():
    return {"message": "Hello from Quiz Fighter Backend!"}

@app.get("/generate-quiz")
async def generate_quiz(user_query: str):
    """
    Generate quiz questions based on a user query using Wikipedia content.
    
    Args:
        user_query: The search query to find relevant Wikipedia content
        
    Returns:
        A JSON object containing quiz questions generated from the Wikipedia content
    """
    # Get Wikipedia content using wiki_search_with_claude
    wiki_content = wiki_search_with_claude(user_query)
    
    # Generate quiz questions from the Wikipedia content
    quiz_data = await generate_quiz_questions(wiki_content)
    
    return quiz_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 