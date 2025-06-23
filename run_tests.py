#!/usr/bin/env python3
"""
Run performance tests for multiple models from config file.
"""

import json
import subprocess
import sys
from pathlib import Path

def load_config(config_file="models_config.json"):
    """Load model configuration."""
    with open(config_file, 'r') as f:
        return json.load(f)

def run_test_for_model(model_name, max_prompts=None):
    """Run performance test for a single model."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    cmd = ["python", "measure_performance.py", model_name]
    
    if max_prompts:
        cmd.extend(["--max-prompts", str(max_prompts)])
    
    # Generate output filename based on model name
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    output_file = f"performance_{safe_model_name}.csv"
    cmd.extend(["--output", output_file])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Successfully tested {model_name}")
            print(f"  Output saved to: results/{output_file}")
        else:
            print(f"✗ Failed to test {model_name}")
            print(f"  Error: {result.stderr}")
    except Exception as e:
        print(f"✗ Exception testing {model_name}: {e}")

def main():
    config = load_config()
    models = config["models"]
    max_prompts = config["test_settings"].get("max_prompts_per_model")
    
    print(f"Running performance tests for {len(models)} models")
    print(f"Max prompts per model: {max_prompts or 'All'}")
    
    # Ensure directories exist
    Path("results").mkdir(exist_ok=True)
    
    successful_models = []
    failed_models = []
    
    for model in models:
        try:
            run_test_for_model(model, max_prompts)
            successful_models.append(model)
        except Exception as e:
            print(f"✗ Failed to test {model}: {e}")
            failed_models.append(model)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully tested: {len(successful_models)} models")
    print(f"Failed: {len(failed_models)} models")
    
    if successful_models:
        print(f"\nSuccessful models:")
        for model in successful_models:
            print(f"  - {model}")
    
    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")
    
    # Run analysis on all successful results
    if successful_models:
        print(f"\nRunning combined analysis...")
        result_files = []
        for model in successful_models:
            safe_model_name = model.replace("/", "_").replace(":", "_")
            result_files.append(f"results/performance_{safe_model_name}.csv")
        
        try:
            cmd = ["python", "analyze_results.py"] + result_files
            subprocess.run(cmd, check=True)
            print("✓ Analysis completed successfully")
        except Exception as e:
            print(f"✗ Analysis failed: {e}")

if __name__ == "__main__":
    main()