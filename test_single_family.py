#!/usr/bin/env python3
"""
Test script that uses family test prompts instead of all prompts.
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

from openai import OpenAI
from tqdm import tqdm

# Import the PerformanceMeasurer class
from measure_performance import PerformanceMeasurer

def load_family_test_prompts() -> List[Dict[str, Any]]:
    """Load family test prompts."""
    prompts = []
    test_dir = Path("test_prompts")
    
    if (test_dir / "family_test_prompts.json").exists():
        with open(test_dir / "family_test_prompts.json", 'r') as f:
            prompts.extend(json.load(f))
    
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Test one prompt from each family")
    parser.add_argument("model", help="OpenRouter model name")
    parser.add_argument("--output", default="family_test_results.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    
    # Initialize measurer
    measurer = PerformanceMeasurer("", args.model)
    api_key = measurer.load_api_key()
    measurer = PerformanceMeasurer(api_key, args.model)
    
    # Load family test prompts
    prompts = load_family_test_prompts()
    if not prompts:
        print("No family test prompts found! Run test_families.py first.")
        sys.exit(1)
    
    print(f"Testing {len(prompts)} family representative prompts")
    
    # Run tests
    measurer.run_all_tests(prompts)
    
    # Save results
    measurer.save_results(args.output)

if __name__ == "__main__":
    main()