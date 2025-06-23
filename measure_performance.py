#!/usr/bin/env python3
"""
Measure performance of OpenRouter models across various tasks.
"""

import argparse
import json
import time
import csv
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import sys
from dotenv import load_dotenv, find_dotenv

# This searches up the directory tree until it finds a .env file
load_dotenv(find_dotenv())

from openai import OpenAI
from tqdm import tqdm

class PerformanceMeasurer:
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/neelnanda/performance-measurement",
                "X-Title": "Performance Measurement Tool"
            }
        )
        self.model_name = model_name
        self.results = []
    
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
    
    def measure_single_prompt(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Measure performance for a single prompt."""
        messages = self.prepare_messages(prompt_data)
        
        # Calculate actual input tokens (more accurate estimate)
        input_text = prompt_data['prompt']
        estimated_input_tokens = self.estimate_tokens(input_text)
        
        try:
            start_time = time.time()
            
            # Make API call with high max_tokens
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100000,  # Set very high for long outputs
                temperature=0.7,
                stream=True  # Enable streaming to measure time to first token
            )
            
            first_token_time = None
            response_text = ""
            chunk_times = []
            
            for chunk in response:
                current_time = time.time()
                
                if chunk.choices[0].delta.content is not None:
                    if first_token_time is None:
                        first_token_time = current_time
                    
                    response_text += chunk.choices[0].delta.content
                    chunk_times.append(current_time)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            time_to_first_token = first_token_time - start_time if first_token_time else 0
            
            # Estimate output tokens
            actual_output_tokens = self.estimate_tokens(response_text)
            actual_output_tokens = max(actual_output_tokens, 1)  # Avoid division by zero
            
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
                'success': True,
                'error': None,
                'timestamp': int(time.time())
            }
            
            # Add reasoning tokens if available (though most models don't expose this)
            result['reasoning_tokens'] = 0  # Placeholder
            
            print(f"✓ {prompt_data.get('id', 'unknown')}: {time_to_first_token:.3f}s first token, "
                  f"{throughput:.1f} tok/s, {total_time:.2f}s total")
            
            return result
            
        except Exception as e:
            print(f"✗ {prompt_data.get('id', 'unknown')}: Error - {str(e)}")
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
                'error': str(e),
                'timestamp': int(time.time())
            }
    
    def run_all_tests(self, prompts: List[Dict[str, Any]], 
                     max_prompts: Optional[int] = None) -> None:
        """Run performance tests on all prompts."""
        if max_prompts:
            prompts = prompts[:max_prompts]
        
        print(f"Testing {len(prompts)} prompts with model: {self.model_name}")
        print("=" * 60)
        
        for prompt_data in tqdm(prompts, desc="Running tests"):
            result = self.measure_single_prompt(prompt_data)
            self.results.append(result)
            
            # Delay between requests for rate limiting (4 requests/min = 15s between requests)
            time.sleep(20)
    
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

def main():
    parser = argparse.ArgumentParser(description="Measure OpenRouter model performance")
    parser.add_argument("model", help="OpenRouter model name (e.g., google/gemini-2.0-flash-exp:free)")
    parser.add_argument("--max-prompts", type=int, help="Maximum number of prompts to test")
    parser.add_argument("--output", help="Output CSV filename (default: auto-generated with model name + timestamp)")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "_").replace(":", "_")
        args.output = f"{model_safe}_{timestamp}.csv"
    
    # Initialize measurer
    measurer = PerformanceMeasurer("", args.model)
    api_key = measurer.load_api_key()
    measurer = PerformanceMeasurer(api_key, args.model)
    
    # Load prompts
    prompts = load_all_prompts()
    if not prompts:
        print("No prompts found! Run generate_prompts.py and generate_images.py first.")
        sys.exit(1)
    
    # Sort prompts by expected complexity (simple first for testing)
    def prompt_complexity(p):
        return p.get('expected_output_tokens', 0) + p.get('input_tokens', 0)
    
    prompts.sort(key=prompt_complexity)
    
    # Run tests
    measurer.run_all_tests(prompts, args.max_prompts)
    
    # Save results
    measurer.save_results(args.output)

if __name__ == "__main__":
    main()