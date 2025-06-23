#!/usr/bin/env python3
"""
Debug script to test what's causing Gemini 2.5 Pro failures.
"""

import asyncio
import aiohttp
import json
from measure_performance_async import AsyncPerformanceMeasurer, load_all_prompts

async def test_single_task(task_id: str, model: str = "google/gemini-2.5-pro"):
    """Test a single problematic task to see exact error."""
    
    # Get API key
    measurer = AsyncPerformanceMeasurer("", model, 1)
    api_key = measurer.load_api_key()
    measurer = AsyncPerformanceMeasurer(api_key, model, 1)
    
    # Load prompts
    prompts = load_all_prompts()
    target_prompt = None
    for prompt in prompts:
        if prompt['id'] == task_id:
            target_prompt = prompt
            break
    
    if not target_prompt:
        print(f"Task {task_id} not found")
        return
    
    print(f"Testing task: {task_id}")
    print(f"Prompt length: {len(target_prompt['prompt']):,} chars")
    print(f"Estimated input tokens: {target_prompt.get('input_tokens', 0):,}")
    print(f"Expected output tokens: {target_prompt.get('expected_output_tokens', 0):,}")
    print("-" * 50)
    
    # Create aiohttp session
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            result = await measurer.measure_single_prompt_with_retry(session, target_prompt)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Exception: {e}")

async def main():
    """Test problematic tasks."""
    problem_tasks = [
        'number_list_10k_1',
        'repeat_after_me_50000_1', 
        'repeat_after_me_10000_1'  # Try a smaller one first
    ]
    
    for task_id in problem_tasks:
        await test_single_task(task_id)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())