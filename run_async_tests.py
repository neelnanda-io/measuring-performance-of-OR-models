#!/usr/bin/env python3
"""
Convenience script to run async performance tests for multiple models.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

async def run_model_test(model_name: str, max_prompts: int = None, max_concurrent: int = 5):
    """Run async test for a single model."""
    output_filename = f"{model_name.replace('/', '_').replace(':', '_')}_results.csv"
    
    cmd = [
        "python", "measure_performance_async.py",
        model_name,
        "--output", output_filename,
        "--max-concurrent", str(max_concurrent)
    ]
    
    if max_prompts:
        cmd.extend(["--max-prompts", str(max_prompts)])
    
    print(f"Starting test for {model_name}...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Completed {model_name}")
        return output_filename
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed {model_name}: {e}")
        print(f"Error output: {e.stderr}")
        return None

def main():
    """Test multiple models and generate comparative analysis."""
    
    # Check API key before starting
    from pathlib import Path
    import os
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        secrets_file = Path("api.secrets")
        if secrets_file.exists():
            with open(secrets_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENROUTER_API_KEY='):
                        api_key = line.split('=', 1)[1].strip('"\'')
                        break
    
    if not api_key or api_key.strip() == "your_actual_api_key_here":
        print("❌ Error: Invalid API key!")
        print("Please edit api.secrets and replace 'your_actual_api_key_here' with your real OpenRouter API key")
        print("You can get an API key from: https://openrouter.ai/keys")
        sys.exit(1)
    
    # Models to test
    models = [
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.5-flash-lite-preview-06-17"
    ]
    
    # Configuration
    max_prompts = 20  # Limit for testing
    max_concurrent = 3  # Conservative to avoid rate limits
    
    print("🚀 Running async performance tests for multiple models")
    print(f"Models: {', '.join(models)}")
    print(f"Max prompts per model: {max_prompts}")
    print(f"Max concurrent requests: {max_concurrent}")
    print("=" * 60)
    
    # Run tests for each model
    result_files = []
    for model in models:
        result_file = asyncio.run(run_model_test(model, max_prompts, max_concurrent))
        if result_file:
            result_files.append(f"results/{result_file}")
    
    # Generate comparative analysis
    if len(result_files) > 0:
        print("\n📊 Generating comparative analysis...")
        try:
            cmd = ["python", "analyze_results.py"] + result_files
            subprocess.run(cmd, check=True)
            print("✓ Analysis complete!")
            print("📁 Check the 'analysis' directory for results")
        except subprocess.CalledProcessError as e:
            print(f"✗ Analysis failed: {e}")
    else:
        print("❌ No successful test results to analyze")

if __name__ == "__main__":
    main()