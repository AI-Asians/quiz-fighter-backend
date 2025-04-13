from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from wiki import wiki_search_with_claude
from generate_questions import generate_quiz_questions
from loadpdf import pdf_search
import asyncio
import os
import shutil
from typing import Optional

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

@app.post("/generate-quiz")
async def generate_quiz(
    user_query: Optional[str] = Form(None),
    pdf_file: Optional[UploadFile] = File(None)
):
    """
    Generate quiz questions based on a PDF file or a user query.
    
    Args:
        user_query: The search query to find relevant Wikipedia content
        pdf_file: Optional PDF file to generate questions from
        
    Returns:
        A JSON object containing quiz questions
    """
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
        content = wiki_search_with_claude(user_query)
    elif not content and not user_query:
        return {"error": "Please provide either a PDF file or a search query"}
    
    # Generate quiz questions from the content
    quiz_data = await generate_quiz_questions(content)
    
    return quiz_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 