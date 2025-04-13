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
    
    # 1) Reduce the context length by ~1/3
    reduced_context_length = int(len(context) * 2/3)
    reduced_context = context[:reduced_context_length]
    
    # 2) Split the text into n parts
    part_length = len(reduced_context) // n if n else len(reduced_context)
    parts = []
    
    for i in range(n):
        start = i * part_length
        end = start + part_length if i < n - 1 else len(reduced_context)
        parts.append(reduced_context[start:end])
    
    # 3) Limit total questions to <= 10
    max_total = 10
    if n * sub_n > max_total:
        parts_needed = max_total // sub_n
        if parts_needed < 1:
            parts_needed = 1
            sub_n = max_total
        if parts_needed < n:
            selected_indices = random.sample(range(n), parts_needed)
            selected_indices.sort()
            parts = [parts[i] for i in selected_indices]
            n = parts_needed
    
    async def generate_questions_for_part(part_index: int, part_text: str) -> List[Dict[str, Any]]:
        prompt = f"""
You are an expert in creating educational quiz questions.

Below is a section of text. Create {sub_n} diverse quiz questions based ONLY on the information in this text.

For each question:
- Some multiple choice (4 options) and some true/false
- The question should be very short and concise
- For multiple choice, the answer choices should only be 1-2 words
- Each question a different concept/fact from the text
- Add a difficulty rating to each question that has to be one of easy, medium, or hard
- For multiple choice, show 4 distinct answer choices and the correct one
- For true/false, indicate if statement is true or false
- Do NOT include any markdown or latex formatting in question or answer choices
- Make the questions funny and trivia-like where possible

IMPORTANT JSON REQUIREMENTS:
- Output valid JSON, with **no additional keys** beyond the sample structure.
- Double quotes inside string values must be escaped (use `\\\"`).
- Do not include any markdown formatting or code blocks. Just raw JSON.

TEXT SECTION:
{part_text}

Output your response **exactly** in the following JSON format:
{{
    "questions": [
        {{
            "question_number": 1,
            "question": "The question text",
            "question_type": "multiple_choice",
            "choices": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "The correct answer",
            "explanation": "short explanation for answer",
            "difficulty": "easy"
        }},
        {{
            "question_number": 2,
            "question": "The question text",
            "question_type": "true_false",
            "correct_answer": "true",
            "explanation": "short explanation for answer",
            "difficulty": "medium"
        }}
    ]
}}
"""
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                system="You must respond with valid JSON matching the specified format exactly."
            )

            
            # Parse the JSON response
            try:
                data = json.loads(response.content[0].text)
                questions = data.get("questions", [])
                
                # Re-index questions
                start_num = part_index * sub_n + 1
                for i, q in enumerate(questions):
                    q["question_number"] = start_num + i
                
                return questions
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response for part {part_index}: {str(e)}")
                return []
        
        except Exception as e:
            print(f"Error generating questions for part {part_index}: {str(e)}")
            return []
    
    # Create and run tasks concurrently
    tasks = [generate_questions_for_part(i, part) for i, part in enumerate(parts)]
    results = await asyncio.gather(*tasks)
    
    # Flatten results and return
    all_questions = []
    for qs in results:
        all_questions.extend(qs)
    
    return {
        "total_questions": len(all_questions),
        "questions": all_questions
    }

# Optional main() for quick test
if __name__ == "__main__":
    async def main():
        test_context = "Lorem ipsum " * 1000  # Fake large text
        data = await generate_quiz_questions(context=test_context, n=10, sub_n=3)
        print(f"Got {data['total_questions']} questions:")
        for q in data["questions"]:
            print(" -", q["question"])
    
    asyncio.run(main())
