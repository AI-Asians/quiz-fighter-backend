import asyncio
import json
from typing import List, Dict, Any, Optional, Literal
from anthropic import AsyncAnthropic, Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

async def generate_quiz_questions(
    context: str,
    n: int = 10,
    sub_n: int = 3,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate quiz questions from a large body of text using Claude Haiku.
    
    Args:
        context: The large body of text to generate questions from
        n: Number of parts to divide the text into (default 10)
        sub_n: Number of questions to generate for each part (default 3)
        api_key: Anthropic API key (optional)
    
    Returns:
        A JSON object containing all generated quiz questions
    """
    # Initialize Anthropic client
    client = AsyncAnthropic(api_key=api_key)
    model = "claude-3-haiku-20240307"
    
    # Define the structure for quiz questions using a tool
    tools = [
        {
            "name": "create_quiz_questions",
            "description": "Generate quiz questions based on the given text",
            "input_schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "Array of quiz questions",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question_number": {
                                    "type": "integer",
                                    "description": "The number of the question"
                                },
                                "question": {
                                    "type": "string",
                                    "description": "The question text"
                                },
                                "question_type": {
                                    "type": "string",
                                    "enum": ["multiple_choice", "true_false"],
                                    "description": "Type of question: multiple_choice or true_false"
                                },
                                "choices": {
                                    "type": "array",
                                    "description": "Answer choices (for multiple_choice only)",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "correct_answer": {
                                    "type": "string",
                                    "description": "For multiple_choice: the correct answer text; For true_false: 'True' or 'False'"
                                },
                                "explanation": {
                                    "type": "string",
                                    "description": "Explanation for the correct answer"
                                }
                            },
                            "required": ["question_number", "question", "question_type", "correct_answer", "explanation"]
                        }
                    }
                },
                "required": ["questions"]
            }
        }
    ]
    
    # Divide the text into n parts
    part_length = len(context) // n
    parts = []
    
    for i in range(n):
        start = i * part_length
        end = start + part_length if i < n - 1 else len(context)
        parts.append(context[start:end])
    
    # Define the prompt template for generating quiz questions
    async def generate_questions_for_part(part_index: int, part_text: str) -> List[Dict[str, Any]]:
        prompt = f"""
        You are an expert in creating educational quiz questions.
        
        Below is a section of text. Create {sub_n} quiz questions based ONLY on the information in this text.
        
        For each question:
        - Create a mix of multiple choice (with 4 options) and true/false questions
        - For multiple choice: provide 4 answer choices and indicate the correct one
        - For true/false: indicate whether the statement is true or false
        - Provide a brief explanation for the correct answer
        - Only create questions that can be definitively answered from the text
        
        TEXT SECTION:
        {part_text}
        
        Remember to ONLY create questions where the answer can be found directly in the text.
        Use the create_quiz_questions tool to format your response.
        """
        
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2000,
                tools=tools,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract the tool use content from the response
            for content in response.content:
                if content.type == "tool_use" and content.name == "create_quiz_questions":
                    questions = content.input.get("questions", [])
                    
                    # Update question numbers to be global rather than per-part
                    start_num = part_index * sub_n + 1
                    for i, q in enumerate(questions):
                        q["question_number"] = start_num + i
                        
                        # Ensure multiple choice questions have choices field
                        if q["question_type"] == "multiple_choice" and "choices" not in q:
                            # If no choices field, create a default one with placeholder
                            q["choices"] = ["Error: Choices not generated properly"]
                    
                    return questions
            
            # If no tool use was found, return an empty list
            return []
            
        except Exception as e:
            print(f"Error generating questions for part {part_index}: {str(e)}")
            return []
    
    # Create tasks for concurrent execution
    tasks = []
    for i, part in enumerate(parts):
        task = generate_questions_for_part(i, part)
        tasks.append(task)
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Flatten the results
    all_questions = []
    for questions in results:
        all_questions.extend(questions)
    
    # Create the final JSON object
    quiz_data = {
        "total_questions": len(all_questions),
        "questions": all_questions
    }
    
    return quiz_data


if __name__ == "__main__":
    asyncio.run(main())