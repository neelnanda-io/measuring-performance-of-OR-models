#!/usr/bin/env python3
"""
Test one prompt from each family to verify system works correctly.
"""

import json
import subprocess
import sys
from pathlib import Path

def get_first_prompt_per_family():
    """Get the first (shortest) prompt from each family."""
    prompts_dir = Path("prompts")
    images_dir = Path("images")
    
    # Load text prompts
    family_prompts = {}
    if (prompts_dir / "all_prompts.json").exists():
        with open(prompts_dir / "all_prompts.json", 'r') as f:
            all_prompts = json.load(f)
            
        for prompt in all_prompts:
            family = prompt["type"]
            if family not in family_prompts:
                family_prompts[family] = prompt
    
    # Load vision prompts
    if (images_dir / "vision_prompts.json").exists():
        with open(images_dir / "vision_prompts.json", 'r') as f:
            vision_prompts = json.load(f)
            
        for prompt in vision_prompts:
            family = prompt["type"]
            if family not in family_prompts:
                family_prompts[family] = prompt
    
    return list(family_prompts.values())

def create_test_prompts_file(selected_prompts):
    """Create a test file with selected prompts."""
    test_dir = Path("test_prompts")
    test_dir.mkdir(exist_ok=True)
    
    with open(test_dir / "family_test_prompts.json", 'w') as f:
        json.dump(selected_prompts, f, indent=2)
    
    return test_dir / "family_test_prompts.json"

def main():
    print("Testing one prompt from each family...")
    
    # Get representative prompts
    selected_prompts = get_first_prompt_per_family()
    print(f"Found {len(selected_prompts)} prompt families:")
    
    for prompt in selected_prompts:
        expected_input = prompt.get('input_tokens', 'unknown')
        expected_output = prompt.get('expected_output_tokens', 'unknown')
        print(f"  - {prompt['type']}: {expected_input} input, {expected_output} expected output tokens")
    
    # Create test file
    test_file = create_test_prompts_file(selected_prompts)
    
    # Temporarily modify the measure_performance.py to use our test file
    print(f"\nCreated test file: {test_file}")
    print("Run the test with:")
    print(f"python test_single_family.py google/gemini-2.0-flash-exp:free")

if __name__ == "__main__":
    main()