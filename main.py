from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from wiki import wiki_search_with_claude
from generate_questions import generate_quiz_questions
from loadpdf import pdf_search
from game_utils import *
import asyncio
import os
import shutil
from typing import Optional

app = FastAPI()
# Dependency to get Supabase client
def get_supabase():
    return initialize_supabase()

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
async def generate_quiz(
    user_query: Optional[str] = Query(None),
    pdf_file: Optional[UploadFile] = File(None),
    device: str = Query(..., description="Device type (web or mobile)")
):
    """
    Generate quiz questions based on a PDF file or a user query.
    
    Args:
        user_query: The search query to find relevant Wikipedia content
        pdf_file: Optional PDF file to generate questions from
        device: Device type ('web' or 'mobile')
        
    Returns:
        A JSON object containing quiz questions and game code
    """
    # Validate device parameter
    if device not in ["web", "mobile"]:
        return {"error": "Device parameter must be either 'web' or 'mobile'"}
    
    content = ""
    temp_pdf_path = None
    
    # If PDF file is provided, save and process it
    if pdf_file and pdf_file.filename.lower().endswith('.pdf'):
        try:
            # Create temp directory if needed
            os.makedirs("temp", exist_ok=True)
            
            # Save the uploaded file
            temp_pdf_path = f"temp/{pdf_file.filename}"
            with open(temp_pdf_path, "wb") as buffer:
                shutil.copyfileobj(pdf_file.file, buffer)
            
            # Extract text from the PDF
            pdf_content = pdf_search(temp_pdf_path)
            
            # If we got content from the PDF, use it
            if pdf_content:
                content = pdf_content
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
        finally:
            # Clean up the temp file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    
    # If no content from PDF, use Wikipedia if query is provided
    if not content and user_query:
        content = await wiki_search_with_claude(user_query)
    elif not content and not user_query:
        return {"error": "Please provide either a PDF file or a search query"}
    
    # Run tasks concurrently:
    # 1. Generate quiz questions
    # 2. Generate theme summary
    
    quiz_task = asyncio.create_task(generate_quiz_questions(content))
    theme_task = asyncio.create_task(generate_theme_summary(content, is_pdf=bool(pdf_file)))
    
    # Wait for both tasks to complete
    quiz_data = await quiz_task
    theme_summary = await theme_task
    
    # Get Supabase client
    supabase_client = initialize_supabase()
    
    matched_questions = await match_questions_with_games(
        quiz_data["questions"], 
        device, 
        supabase_client
    )
    
    # Update game configs based on theme
    updated_questions = await update_game_configs(matched_questions, theme_summary)
    
    # Check if we have any matched questions
    if not updated_questions:
        return {
            "error": f"No matching games found for device type '{device}' and the generated questions."
        }
    
    # Prepare response - now using the number of matched questions
    response = {
        "total_questions": len(updated_questions),
        "questions": updated_questions,
        "theme_summary": theme_summary
    }
    
    response["questions"] = response["questions"][:3]
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
