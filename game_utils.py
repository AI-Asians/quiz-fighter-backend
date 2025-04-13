# game_utils.py
import json
import re
import os
import time
import logging
import sys
import random
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv

import anthropic
import supabase
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

load_dotenv()

# Set up logging (simple format)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Initialize Anthropic client (sync client, we'll make calls in a thread)
claude_client = anthropic.Anthropic()

def initialize_supabase():
    """
    Initialize Supabase client
    """
    start_time = time.time()
    logger.info("Initializing Supabase client")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        logger.error("Missing Supabase environment variables")
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

    client = supabase.create_client(url, key)

    end_time = time.time()
    logger.info(f"Supabase client initialized in {end_time - start_time:.2f} seconds")
    return client


async def generate_theme_summary(content: str, is_pdf: bool = False) -> str:
    """
    Generate a short 1-2 sentence theme summary using Claude.
    """
    logger.info("Starting theme summary generation")
    start_time = time.time()

    prompt = f"""
Please provide a 2-3 sentence summary of the main theme of the following text.
Focus on capturing the core subject matter and key concepts.

TEXT:
{content}
"""

    try:
        # Wrap the synchronous call in asyncio.to_thread for real parallelization
        def sync_claude_call():
            return claude_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=150,
                temperature=0.2,
                system="You are a helpful assistant that provides concise and accurate thematic summaries.",
                messages=[{"role": "user", "content": prompt}]
            )

        response = await asyncio.to_thread(sync_claude_call)
        end_time = time.time()
        logger.info(f"Theme summary generation completed in {end_time - start_time:.2f} seconds")

        # Access the text returned by Claude
        # The response structure depends on how the anthropic client library returns data
        if response and response.content:
            return response.content[0].text  # or your equivalent text extraction
        return "Unable to generate theme summary."

    except Exception as e:
        end_time = time.time()
        logger.error(f"Error generating theme summary: {str(e)} (took {end_time - start_time:.2f} seconds)")
        return "Unable to generate theme summary."


async def match_questions_with_games(questions: List[dict], device: str, supabase_client) -> List[dict]:
    """
    Match each question with a game in Supabase (based on metadata).
    Run queries in parallel for speed (if needed).
    """
    start_time = time.time()
    logger.info(f"Starting to match {len(questions)} questions with games")

    match_tasks = []
    for q in questions:
        if not isinstance(q, dict):
            # Safety check if somehow the question is a string
            continue
        match_tasks.append(asyncio.create_task(match_question_with_game(q, device, supabase_client)))

    matched_questions = await asyncio.gather(*match_tasks)
    end_time = time.time()

    matched_count = sum(1 for mq in matched_questions if mq.get("game_id") is not None)
    logger.info(f"Matched {matched_count}/{len(questions)} questions with games in {end_time - start_time:.2f} seconds")

    return matched_questions


async def match_question_with_game(question: dict, device: str, supabase_client) -> dict:
    """
    Match a single question with a game from Supabase. 
    """
    q_id = question.get("id", "unknown")
    start_time = time.time()
    logger.info(f"Starting to match question {q_id}")

    # Example question type usage
    question_type = question.get("question_type", "multiple_choice")

    try:
        logger.info(f"Querying Supabase for question {q_id}")
        db_start_time = time.time()
        # For big tables, apply an actual filter; here it's simplified
        response = supabase_client.table("game_data").select("*").execute()
        db_end_time = time.time()
        logger.info(f"Supabase query completed in {db_end_time - db_start_time:.2f} seconds for question {q_id}")

        matched_games = []
        for game in response.data or []:
            metadata = game.get("metadata", {})
            if metadata.get("device") == device and metadata.get("question_type") == question_type:
                matched_games.append(game)

        if matched_games:
            selected = random.choice(matched_games)
            question["game_id"] = selected["id"]
            question["original_config"] = selected["config"]
            question["original_code"] = selected["code"]
            logger.info(
                f"Successfully matched question {q_id} with game {selected['id']} "
                f"in {time.time() - start_time:.2f} seconds"
            )
        else:
            question["game_id"] = None
            question["original_config"] = None
            question["original_code"] = None
            logger.info(f"No matching game found for question {q_id}")

    except Exception as e:
        logger.error(f"Error matching question {q_id} with game: {str(e)}")
        question["game_id"] = None
        question["original_config"] = None
        question["original_code"] = None

    return question


async def update_game_configs(questions: List[dict], theme_summary: str) -> List[dict]:
    """
    For each question that has a matching game/config, update the config in parallel.
    """
    start_time = time.time()
    total_questions = len(questions)
    eligible = [q for q in questions if q.get("game_id") and q.get("original_config")]

    logger.info(f"Starting to update configs for {len(eligible)}/{total_questions} eligible questions")

    tasks = []
    for i, question in enumerate(eligible):
        tasks.append(asyncio.create_task(process_question_config(question, theme_summary, i)))

    if not tasks:
        return []

    logger.info(f"Waiting for {len(tasks)} config update tasks to complete")
    updated_questions = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(
        f"Updated {len(updated_questions)} configs in {total_time:.2f} seconds "
        f"(avg: {total_time / len(updated_questions):.2f}s per question)"
    )
    return updated_questions


async def process_question_config(question: dict, theme_summary: str, index: int) -> dict:
    """
    Process a single question's config:
    1. Make a Claude call to update the config
    2. Replace config in the code
    """
    q_id = question.get("id", f"unknown-{index}")
    start_time = time.time()
    logger.info(f"Processing config for question {q_id} (index: {index})")

    # 1) Update config
    updated_config = await update_config_with_theme(
        original_config=question["original_config"],
        theme_summary=theme_summary,
        question=question,
        q_id=q_id
    )

    # 2) Replace in code
    original_code = question.get("original_code", "")
    if updated_config and original_code:
        updated_code = replace_config_in_code(original_code, updated_config, q_id)
        question["code"] = updated_code

    # Clean up
    question.pop("original_config", None)
    question.pop("original_code", None)

    logger.info(f"Finished processing question {q_id} in {time.time() - start_time:.2f} seconds")
    return question


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type((
        Exception,
        anthropic.APIError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def update_config_with_theme(
    original_config: str,
    theme_summary: str,
    question: dict,
    q_id: str = "unknown"
) -> str:
    """
    Call Claude to update a game's JS config with the given theme_summary and question data.
    Runs the synchronous call in a background thread for true async concurrency.
    """
    start_time = time.time()
    logger.info(f"Updating config for question {q_id}")

    # Build a simple text prompt (removed the structured "tools" complexity)
    question_str = "\n".join(f"{k}: {v}" for k, v in question.items() if k not in ["original_config", "original_code", "code"])
    prompt = f"""
You are a helpful assistant that updates a JavaScript config to reflect a specific theme and question details.

THEME SUMMARY:
{theme_summary}

QUESTION CONTENT:
{question_str}

ORIGINAL CONFIG:
{original_config}

Please modify color schemes, text elements, or visual components to match the theme, while preserving the structure.
Include the declaration: const config = {{ ... }};
"""

    # Do the synchronous Anthrop ic call in a thread
    def sync_claude_call():
        return claude_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2000,
            temperature=0,
            system="You update JS config based on themes.",
            messages=[{"role": "user", "content": prompt}]
        )

    try:
        response = await asyncio.to_thread(sync_claude_call)
        if not response or not response.content:
            logger.warning(f"No response from Claude for question {q_id}. Returning original config.")
            return original_config

        # Attempt to parse the text. In your real usage, confirm the exact structure
        new_config_text = response.content[0].text

        logger.info(f"Config update for question {q_id} done in {time.time() - start_time:.2f} seconds")
        return new_config_text.strip() if new_config_text else original_config

    except Exception as e:
        logger.error(f"Error updating config for question {q_id}: {str(e)}")
        # Raise again so tenacity can retry if needed
        raise


def replace_config_in_code(original_code: str, updated_config: str, q_id: str = "unknown") -> str:
    """
    Replace the existing 'const config = {...}' in original_code with updated_config.
    """
    logger.info(f"Replacing config in code for question {q_id}")
    pattern = r'const\s+config\s*=\s*(\{[\s\S]*?\})\s*;'
    match = re.search(pattern, original_code)
    if not match:
        logger.warning(f"No 'const config' found in original code for question {q_id}")
        return original_code

    # Ensure the updated_config itself has 'const config = {...};'
    if not updated_config.strip().startswith("const config"):
        # Attempt to extract from updated_config if it includes braces
        match2 = re.search(pattern, updated_config)
        if match2:
            updated_config = f"const config = {match2.group(1)};"
        else:
            # fallback: wrap everything
            updated_config = "const config = " + updated_config.strip()
            if not updated_config.endswith(";"):
                updated_config += ";"

    # Now replace
    try:
        new_code = re.sub(pattern, updated_config, original_code, count=1)
        return new_code
    except Exception as e:
        logger.error(f"Error replacing config in code for question {q_id}: {str(e)}")
        return original_code
