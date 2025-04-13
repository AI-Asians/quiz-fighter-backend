import os
import io
from typing import List, Optional, Dict, Any
import anthropic
import json
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader  # Use pypdf instead of PyMuPDF

def pdf_search(file_path: str) -> str:
    """
    Function that:
    1. Takes a PDF file path
    2. Extracts text from the PDF
    3. Returns the text content for quiz generation
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        A single string containing all extracted PDF content
    """
    # Step 1: Check if file exists
    if not os.path.exists(file_path):
        print(f"PDF file not found: {file_path}")
        return ""
    
    # Step 2: Extract text from PDF
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        print("No text extracted from PDF")
        return ""
    
    # Step 3: Process the text (simple approach, no need for Claude here)
    processed_text = raw_text
    
    return processed_text

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using pypdf.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        text = ""
        # Open the PDF file
        with open(file_path, "rb") as file:
            # Create a PDF reader object
            pdf_reader = PdfReader(file)
            
            # Extract text from each page
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

# Simple function to install the required dependency if it's missing
def ensure_dependencies():
    try:
        import pypdf
    except ImportError:
        print("Installing required dependencies...")
        import subprocess
        subprocess.check_call(["pip", "install", "pypdf"])
        print("Dependencies installed successfully")

# Make sure dependencies are available on import
ensure_dependencies()