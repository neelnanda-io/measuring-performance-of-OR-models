#!/usr/bin/env python3
"""
Async version of performance measurement script with retry logic.
"""

import argparse
import json
import time
import csv
import base64
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import sys
from tqdm.asyncio import tqdm
from dotenv import load_dotenv, find_dotenv

# This searches up the directory tree until it finds a .env file
load_dotenv(find_dotenv())

class AsyncPerformanceMeasurer:
    def __init__(self, api_key: str, model_name: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.results = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    def load_api_key(self) -> str:
        """Load API key from environment variable or api.secrets file."""
        # First try environment variable
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key:
            return api_key
        
        # Try loading from api.secrets file
        secrets_file = Path("api.secrets")
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('OPENROUTER_API_KEY='):
                            return line.split('=', 1)[1].strip('"\'')
            except Exception as e:
                print(f"Error reading api.secrets: {e}")
        
        print("Error: OPENROUTER_API_KEY not found.")
        print("Please either:")
        print("1. Set environment variable: export OPENROUTER_API_KEY=your_key_here")
        print("2. Create api.secrets file with: OPENROUTER_API_KEY=your_key_here")
        sys.exit(1)
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def estimate_tokens(self, text: str) -> int:
        """
        More accurate token estimation.
        
        OpenAI/GPT tokenizers typically follow these patterns:
        - ~4 characters per token on average
        - Common words are often single tokens
        - Punctuation and special characters can be separate tokens
        """
        if not text:
            return 0
        
        # Basic approach: character count / 4, but with adjustments
        char_based = len(text) / 4
        
        # Word-based adjustment for very short texts
        words = len(text.split())
        word_based = words * 1.3  # Average 1.3 tokens per word
        
        # Use the more conservative estimate for accuracy
        return int(min(char_based, word_based))
    
    def prepare_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare messages for the API call."""
        messages = []
        
        # Handle vision tasks with images
        if 'images' in prompt_data and prompt_data['images']:
            content = [{"type": "text", "text": prompt_data['prompt']}]
            
            for image_path in prompt_data['images']:
                if os.path.exists(image_path):
                    base64_image = self.encode_image(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
            
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            # Text-only task
            messages.append({
                "role": "user", 
                "content": prompt_data['prompt']
            })
        
        return messages
    
    async def make_api_request(self, session: aiohttp.ClientSession, messages: List[Dict[str, Any]], prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make async API request with streaming."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/neelnanda/performance-measurement",
            "X-Title": "Performance Measurement Tool"
        }
        
        # Calculate appropriate max_tokens based on expected output
        expected_output = prompt_data.get('expected_output_tokens', 4000)
        # Cap at reasonable limits to avoid model-specific issues
        max_tokens = min(expected_output * 1.2, 100000)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": 0.7,
            "stream": True
        }
        
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")
            
            # Parse streaming response
            response_text = ""
            first_token_time = None
            start_time = time.time()
            
            async for line in response.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content')
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    response_text += content
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            
            return {
                'response_text': response_text,
                'first_token_time': first_token_time,
                'start_time': start_time,
                'end_time': end_time
            }
    
    async def measure_single_prompt_with_retry(self, session: aiohttp.ClientSession, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Measure performance for a single prompt with retry logic."""
        async with self.semaphore:  # Limit concurrent requests
            messages = self.prepare_messages(prompt_data)
            
            # Calculate actual input tokens (more accurate estimate)
            input_text = prompt_data['prompt']
            estimated_input_tokens = self.estimate_tokens(input_text)
            
            max_retries = 4
            retry_delay = 20  # seconds
            
            for attempt in range(max_retries):
                try:
                    response_data = await self.make_api_request(session, messages, prompt_data)
                    
                    # Calculate metrics
                    start_time = response_data['start_time']
                    end_time = response_data['end_time']
                    first_token_time = response_data['first_token_time']
                    response_text = response_data['response_text']
                    
                    total_time = end_time - start_time
                    time_to_first_token = first_token_time - start_time if first_token_time else 0
                    
                    # Estimate output tokens
                    actual_output_tokens = self.estimate_tokens(response_text)
                    actual_output_tokens = max(actual_output_tokens, 1)
                    
                    # Calculate throughput (tokens per second)
                    # Use generation time (excluding time to first token) for throughput
                    generation_time = end_time - (first_token_time or start_time)
                    
                    # For very fast responses, use total time to avoid inflated numbers
                    if generation_time < 0.1:  # Less than 100ms
                        generation_time = total_time
                    
                    throughput = actual_output_tokens / generation_time if generation_time > 0 else 0
                    
                    result = {
                        'model_name': self.model_name,
                        'prompt_id': prompt_data.get('id', 'unknown'),
                        'prompt_type': prompt_data.get('type', 'unknown'),
                        'input_tokens': int(estimated_input_tokens),
                        'output_tokens': int(actual_output_tokens),
                        'expected_output_tokens': prompt_data.get('expected_output_tokens', 0),
                        'time_to_first_token': round(time_to_first_token, 4),
                        'total_time': round(total_time, 4),
                        'throughput_tokens_per_sec': round(throughput, 2),
                        'response_length_chars': len(response_text),
                        'reasoning_tokens': 0,
                        'success': True,
                        'error': None,
                        'timestamp': int(time.time())
                    }
                    
                    print(f"✓ {prompt_data.get('id', 'unknown')}: {time_to_first_token:.3f}s first token, "
                          f"{throughput:.1f} tok/s, {total_time:.2f}s total")
                    
                    return result
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a rate limit error
                    is_rate_limit = ('429' in error_str or 
                                   'rate limit' in error_str.lower() or 
                                   'rate-limit' in error_str.lower())
                    
                    if is_rate_limit and attempt < max_retries - 1:
                        print(f"⚠️  {prompt_data.get('id', 'unknown')}: Rate limit hit, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        # Final failure or non-rate-limit error
                        # Show more details for debugging
                        if len(error_str) > 200:
                            error_summary = error_str[:200] + "..."
                        else:
                            error_summary = error_str
                        print(f"✗ {prompt_data.get('id', 'unknown')}: Error - {error_summary}")
                        return {
                            'model_name': self.model_name,
                            'prompt_id': prompt_data.get('id', 'unknown'),
                            'prompt_type': prompt_data.get('type', 'unknown'),
                            'input_tokens': int(estimated_input_tokens),
                            'output_tokens': 0,
                            'expected_output_tokens': prompt_data.get('expected_output_tokens', 0),
                            'time_to_first_token': 0,
                            'total_time': 0,
                            'throughput_tokens_per_sec': 0,
                            'response_length_chars': 0,
                            'reasoning_tokens': 0,
                            'success': False,
                            'error': error_str,
                            'timestamp': int(time.time())
                        }
            
            # Should not reach here
            return {
                'model_name': self.model_name,
                'prompt_id': prompt_data.get('id', 'unknown'),
                'prompt_type': prompt_data.get('type', 'unknown'),
                'input_tokens': int(estimated_input_tokens),
                'output_tokens': 0,
                'expected_output_tokens': prompt_data.get('expected_output_tokens', 0),
                'time_to_first_token': 0,
                'total_time': 0,
                'throughput_tokens_per_sec': 0,
                'response_length_chars': 0,
                'reasoning_tokens': 0,
                'success': False,
                'error': 'Max retries exceeded',
                'timestamp': int(time.time())
            }
    
    async def run_all_tests_async(self, prompts: List[Dict[str, Any]], 
                                 max_prompts: Optional[int] = None) -> None:
        """Run performance tests on all prompts asynchronously."""
        if max_prompts:
            prompts = prompts[:max_prompts]
        
        print(f"Testing {len(prompts)} prompts with model: {self.model_name}")
        print(f"Max concurrent requests: {self.max_concurrent}")
        print("=" * 60)
        
        # Create aiohttp session with timeout
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout per request
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Create tasks for all prompts
            tasks = [
                self.measure_single_prompt_with_retry(session, prompt_data)
                for prompt_data in prompts
            ]
            
            # Run tasks with progress bar
            results = []
            for task in tqdm.as_completed(tasks, desc="Running tests"):
                result = await task
                results.append(result)
            
            self.results = results
    
    def save_results(self, filename: str) -> None:
        """Save results to CSV file."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        if not self.results:
            print("No results to save!")
            return
        
        fieldnames = [
            'model_name', 'prompt_id', 'prompt_type',
            'input_tokens', 'output_tokens', 'expected_output_tokens', 'reasoning_tokens',
            'time_to_first_token', 'total_time', 'throughput_tokens_per_sec',
            'response_length_chars', 'success', 'error', 'timestamp'
        ]
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\nResults saved to: {filepath}")
        print(f"Total tests: {len(self.results)}")
        print(f"Successful tests: {sum(1 for r in self.results if r['success'])}")
        print(f"Failed tests: {sum(1 for r in self.results if not r['success'])}")

def load_all_prompts() -> List[Dict[str, Any]]:
    """Load all test prompts from JSON files."""
    prompts = []
    prompts_dir = Path("prompts")
    
    # Load text prompts
    if (prompts_dir / "all_prompts.json").exists():
        with open(prompts_dir / "all_prompts.json", 'r') as f:
            prompts.extend(json.load(f))
    
    # Load vision prompts
    images_dir = Path("images")
    if (images_dir / "vision_prompts.json").exists():
        with open(images_dir / "vision_prompts.json", 'r') as f:
            prompts.extend(json.load(f))
    
    return prompts

async def main():
    parser = argparse.ArgumentParser(description="Async OpenRouter model performance measurement")
    parser.add_argument("model", help="OpenRouter model name (e.g., google/gemini-2.5-flash-lite-preview-06-17)")
    parser.add_argument("--max-prompts", type=int, help="Maximum number of prompts to test")
    parser.add_argument("--output", help="Output CSV filename (default: auto-generated with model name + timestamp)")
    parser.add_argument("--max-concurrent", type=int, default=1000, help="Maximum concurrent requests")
    parser.add_argument("--include-long", action="store_true", help="Include tasks with >50K input or output tokens (warning: very expensive!)")
    
    args = parser.parse_args()
    
    # Load API key first
    temp_measurer = AsyncPerformanceMeasurer("", args.model, args.max_concurrent)
    api_key = temp_measurer.load_api_key()
    
    # Validate API key
    if not api_key or api_key.strip() == "your_actual_api_key_here":
        print("❌ Error: Invalid API key!")
        print("Please edit api.secrets and replace 'your_actual_api_key_here' with your real OpenRouter API key")
        print("You can get an API key from: https://openrouter.ai/keys")
        sys.exit(1)
    
    # Generate output filename if not provided
    if not args.output:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "_").replace(":", "_")
        args.output = f"{model_safe}_{timestamp}.csv"
    
    # Initialize measurer with valid API key
    measurer = AsyncPerformanceMeasurer(api_key, args.model, args.max_concurrent)
    
    # Load prompts
    prompts = load_all_prompts()
    if not prompts:
        print("No prompts found! Run generate_prompts.py and generate_images.py first.")
        sys.exit(1)
    
    # Filter out long tasks unless explicitly requested
    if not args.include_long:
        original_count = len(prompts)
        prompts = [p for p in prompts if not p.get('is_long', False)]
        filtered_count = original_count - len(prompts)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} long tasks (>50K tokens). Use --include-long to include them.")
    
    # Sort prompts by expected complexity (simple first for testing)
    def prompt_complexity(p):
        return p.get('expected_output_tokens', 0) + p.get('input_tokens', 0)
    
    prompts.sort(key=prompt_complexity)
    
    # Run tests
    await measurer.run_all_tests_async(prompts, args.max_prompts)
    
    # Save results
    measurer.save_results(args.output)

if __name__ == "__main__":
    asyncio.run(main())