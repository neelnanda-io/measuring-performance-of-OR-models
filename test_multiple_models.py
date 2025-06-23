#!/usr/bin/env python3
"""
Test multiple models from configuration file.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

def load_models_config(config_file="models_to_test.json"):
    """Load models configuration."""
    with open(config_file, 'r') as f:
        return json.load(f)

def run_test_for_model(model_name, config):
    """Run performance test for a single model."""
    print(f"\n{'='*70}")
    print(f"Testing model: {model_name}")
    print(f"{'='*70}")
    
    # Determine which script to use
    if config["test_settings"].get("use_family_tests_only", False):
        script = "test_single_family.py"
    else:
        script = "measure_performance.py"
    
    cmd = ["python", script, model_name]
    
    max_prompts = config["test_settings"].get("max_prompts_per_model")
    if max_prompts:
        cmd.extend(["--max-prompts", str(max_prompts)])
    
    # Generate output filename based on model name
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    output_file = f"performance_{safe_model_name}.csv"
    cmd.extend(["--output", output_file])
    
    try:
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✓ Successfully tested {model_name}")
            print(f"  Output saved to: results/{output_file}")
            return True
        else:
            print(f"✗ Failed to test {model_name}")
            print(f"  Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout testing {model_name} (exceeded 1 hour)")
        return False
    except Exception as e:
        print(f"✗ Exception testing {model_name}: {e}")
        return False

def main():
    config = load_models_config()
    models = config["models"]
    
    print(f"Running performance tests for {len(models)} models")
    print(f"Configuration: {config['test_settings']}")
    
    # Ensure directories exist
    Path("results").mkdir(exist_ok=True)
    
    successful_models = []
    failed_models = []
    
    for i, model in enumerate(models):
        print(f"\nProgress: {i+1}/{len(models)} models")
        
        try:
            success = run_test_for_model(model, config)
            if success:
                successful_models.append(model)
            else:
                failed_models.append(model)
        except Exception as e:
            print(f"✗ Failed to test {model}: {e}")
            failed_models.append(model)
        
        # Add delay between models to avoid rate limiting
        if i < len(models) - 1:  # Don't delay after the last model
            delay = config["test_settings"].get("delay_between_models", 60)
            print(f"Waiting {delay}s before next model...")
            time.sleep(delay)
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully tested: {len(successful_models)} models")
    print(f"Failed: {len(failed_models)} models")
    
    if successful_models:
        print(f"\nSuccessful models:")
        for model in successful_models:
            print(f"  ✓ {model}")
    
    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  ✗ {model}")
    
    # Run analysis on all successful results
    if successful_models:
        print(f"\nRunning combined analysis...")
        result_files = []
        for model in successful_models:
            safe_model_name = model.replace("/", "_").replace(":", "_")
            result_file = f"results/performance_{safe_model_name}.csv"
            if Path(result_file).exists():
                result_files.append(result_file)
        
        if result_files:
            try:
                cmd = ["python", "analyze_results.py"] + result_files
                subprocess.run(cmd, check=True)
                print("✓ Multi-model analysis completed successfully")
                print("📊 Check analysis/ directory for reports and visualizations")
            except Exception as e:
                print(f"✗ Analysis failed: {e}")
        else:
            print("No result files found for analysis")

if __name__ == "__main__":
    main()