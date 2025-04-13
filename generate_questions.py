import asyncio
import json
from typing import List, Dict, Any, Optional
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import os
import random

load_dotenv()

async def generate_quiz_questions(
    context: str,
    n: int = 8,      # number of parts to split text into
    sub_n: int = 2,   # questions per part
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate quiz questions from a large body of text using Claude Haiku.
    
    Ensures we never generate more than 10 total questions.
    If n*sub_n > 10, we randomly sample only enough parts so that
    total possible questions <= 10.
    """
    client = AsyncAnthropic(api_key=api_key)
    model = "claude-3-haiku-20240307"
    
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
                                "question_number": {"type": "integer"},
                                "question": {"type": "string"},
                                "question_type": {
                                    "type": "string",
                                    "enum": ["multiple_choice", "true_false"]
                                },
                                "choices": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "correct_answer": {"type": "string"},
                                "explanation": {"type": "string"}
                            },
                            "required": [
                                "question_number", "question", "question_type",
                                "correct_answer", "explanation"
                            ]
                        }
                    }
                },
                "required": ["questions"]
            }
        }
    ]
    
    # 1) Reduce the context length by ~1/3 just as you did originally
    reduced_context_length = int(len(context) * 2/3)
    reduced_context = context[:reduced_context_length]
    
    # 2) Split the text into n parts
    part_length = len(reduced_context) // n if n else len(reduced_context)
    parts = []
    
    for i in range(n):
        start = i * part_length
        # last chunk goes to the end of reduced_context
        end = start + part_length if i < n - 1 else len(reduced_context)
        parts.append(reduced_context[start:end])
    
    # --------------------------------------------------------------
    # *** KEY CHANGE TO LIMIT TOTAL QUESTIONS TO <= 10 ***
    # If n * sub_n is more than 10, randomly sample fewer parts
    # so that we cannot exceed 10 total. For example, if sub_n=3
    # we can only handle at most floor(10/3) = 3 parts => 9 questions.
    # --------------------------------------------------------------
    max_total = 10
    if n * sub_n > max_total:
        # Figure out how many parts we can keep so sub_n * parts_needed <= 10
        parts_needed = max_total // sub_n  # integer division
        
        # If sub_n is so large that even 1 part can exceed 10, clamp sub_n
        if parts_needed < 1:
            parts_needed = 1
            sub_n = max_total  # now each part can generate up to 10
        
        # Randomly pick 'parts_needed' parts out of n
        if parts_needed < n:
            selected_indices = random.sample(range(n), parts_needed)
            # Sort them so final ordering is stable
            selected_indices.sort()
            parts = [parts[i] for i in selected_indices]
            n = parts_needed
    # --------------------------------------------------------------
    
    async def generate_questions_for_part(part_index: int, part_text: str) -> List[Dict[str, Any]]:
        prompt = f"""
You are an expert in creating educational quiz questions.

Below is a section of text. Create {sub_n} diverse quiz questions based ONLY on the information in this text.

For each question:
- Some multiple choice (4 options) and some true/false
- Each question a different concept/fact from the text
- For multiple choice, show 4 distinct answer choices and the correct one
- For true/false, indicate if statement is true or false
- Provide an explanation citing the text

TEXT SECTION:
{part_text}

Use the create_quiz_questions tool to format your response.
"""
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2000,
                tools=tools,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract the tool-use block from the response
            for content in response.content:
                if content.type == "tool_use" and content.name == "create_quiz_questions":
                    questions = content.input.get("questions", [])
                    
                    # Re-index questions so they remain unique across parts
                    start_num = part_index * sub_n + 1
                    for i, q in enumerate(questions):
                        q["question_number"] = start_num + i
                        if q["question_type"] == "multiple_choice" and "choices" not in q:
                            q["choices"] = ["Error: Choices not generated properly"]
                    
                    return questions
            
            return []
        
        except Exception as e:
            print(f"Error generating questions for part {part_index}: {str(e)}")
            return []
    
    # 3) Create tasks for each part
    tasks = []
    for i, part in enumerate(parts):
        tasks.append(generate_questions_for_part(i, part))
    
    # 4) Run tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # 5) Flatten into one list
    all_questions = []
    for qs in results:
        all_questions.extend(qs)
    
    # 6) Final result
    quiz_data = {
        "total_questions": len(all_questions),
        "questions": all_questions
    }
    
    return quiz_data

# Optional main() for quick test
if __name__ == "__main__":
    async def main():
        test_context = "Lorem ipsum " * 1000  # Fake large text
        data = await generate_quiz_questions(context=test_context, n=10, sub_n=3)
        print(f"Got {data['total_questions']} questions:")
        for q in data["questions"]:
            print(" -", q["question"])
    
    asyncio.run(main())
