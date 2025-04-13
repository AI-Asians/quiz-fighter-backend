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
                                },
                                "difficulty": {
                                    "type": "string",
                                    "enum": ["easy", "medium", "hard"],
                                    "description": "The difficulty level of the question"
                                },
                                "success_pun": {
                                    "type": "string",
                                    "description": "A short, punny exclamation related to the question to display when answered correctly. MAXIMUM 3 WORDS LONG"
                                }
                            },
                            "required": ["question_number", "question", "question_type", "correct_answer", "explanation", "difficulty", "success_pun"]
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
        
        Below is a section of text. Create {sub_n} diverse quiz questions based ONLY on the information in this text.
        
        For each question:
        - Create a mix of multiple choice (with 4 options) and true/false questions
        - ENSURE EACH QUESTION COVERS A DIFFERENT CONCEPT OR FACT from the text
        - STAY STRICTLY FOCUSED ON THE MAIN TOPIC of the text
        - Identify the primary subject matter in the first paragraph and maintain focus on that subject
        - If the text contains information about multiple topics, only create questions about the dominant topic
        - Focus on KEY CONCEPTS that would appear on an exam about this topic
        - For multiple choice: provide 4 distinct answer choices and indicate the correct one
        - THE CORRECT ANSWER MUST BE AN EXACT MATCH of one of the 4 answer choices
        - Make distractors plausible but clearly incorrect based on the text
        - For true/false: indicate whether the statement is true or false
        - Provide a concise explanation for the correct answer citing the text
        - Questions should test understanding at various levels (recall, application, analysis)
        - Assign a difficulty level (easy, medium, hard) to each question based on how challenging it is
        - Create a short, punny exclamation related to the question content to display when a user answers correctly
        - Questions must be clearly answerable from the provided text
        
        TEXT SECTION:
        {part_text}
        
        IMPORTANT:
        - DOUBLE-CHECK that your correct_answer EXACTLY matches one of the choices provided
        - COMPLETELY IGNORE any portions of text that don't relate to the main topic identified in the first paragraph
        - Do NOT repeat similar questions or test the same concept multiple times
        - Do NOT introduce concepts from outside the provided text
        - DO NOT mix topics or shift between unrelated subjects (like from animals to computer science)
        - VERIFY all answers are properly matched with choices before submitting
        - If you find an inconsistency in your multiple-choice options, fix it before finalizing
        - Make sure each success_pun is relevant and appropriate
        
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
                            
                        # Ensure difficulty is set
                        if "difficulty" not in q:
                            q["difficulty"] = "medium"
                            
                        # Ensure success pun is set
                        if "success_pun" not in q:
                            q["success_pun"] = "Great job!"
                    
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