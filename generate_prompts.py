#!/usr/bin/env python3
"""
Generate diverse test prompts for performance measurement.
"""

import json
import os
import random
import requests
from pathlib import Path

def get_wikipedia_content(title, target_tokens=None):
    """Fetch Wikipedia content for a given title."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        content = data.get('extract', '')
        
        if target_tokens:
            # Rough approximation: 1 token ≈ 0.75 words
            target_words = int(target_tokens * 0.75)
            words = content.split()
            if len(words) > target_words:
                content = ' '.join(words[:target_words])
        
        return content
    except:
        return f"Sample text content for {title}. " * (target_tokens // 10 if target_tokens else 100)

def generate_repetitive_content(word, target_tokens):
    """Generate repetitive content with a specific word."""
    content = (word + " ") * target_tokens
    return content.strip()

def get_simple_dictionary():
    """Get 500 common single-token words."""
    return [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "is", "was", "are", "been", "has", "had", "were", "said", "each", "which",
        "she", "do", "how", "their", "if", "will", "up", "other", "about", "out",
        "many", "then", "them", "these", "so", "some", "her", "would", "make", "like",
        "into", "him", "time", "has", "two", "more", "go", "no", "way", "could",
        "my", "than", "first", "been", "call", "who", "its", "now", "find", "long",
        "down", "day", "did", "get", "come", "made", "may", "part", "over", "new",
        "sound", "take", "only", "little", "work", "know", "place", "year", "live", "me",
        "back", "give", "most", "very", "after", "thing", "our", "name", "good", "sentence",
        "man", "think", "say", "great", "where", "help", "through", "much", "before", "line",
        "right", "too", "mean", "old", "any", "same", "tell", "boy", "follow", "came",
        "want", "show", "also", "around", "form", "three", "small", "set", "put", "end",
        "why", "again", "turn", "here", "off", "went", "old", "number", "great", "tell",
        "men", "say", "small", "every", "found", "still", "between", "name", "should", "home",
        "big", "give", "air", "line", "set", "own", "under", "read", "last", "never",
        "us", "left", "end", "along", "while", "might", "next", "sound", "below", "saw",
        "something", "thought", "both", "few", "those", "always", "looked", "show", "large", "often",
        "together", "asked", "house", "don't", "world", "going", "want", "school", "important", "until",
        "form", "food", "keep", "children", "feet", "land", "side", "without", "boy", "once",
        "animal", "life", "enough", "took", "sometimes", "four", "head", "above", "kind", "began",
        "almost", "live", "page", "got", "earth", "need", "far", "hand", "high", "year",
        "mother", "light", "country", "father", "let", "night", "picture", "being", "study", "second",
        "soon", "story", "since", "white", "ever", "paper", "hard", "near", "sentence", "better",
        "best", "across", "during", "today", "however", "sure", "knew", "it's", "try", "told",
        "young", "sun", "thing", "whole", "hear", "example", "heard", "several", "change", "answer",
        "room", "sea", "against", "top", "turned", "learn", "point", "city", "play", "toward",
        "five", "himself", "usually", "money", "seen", "didn't", "car", "morning", "i'm", "body",
        "upon", "family", "later", "turn", "move", "face", "door", "cut", "done", "group",
        "true", "leave", "color", "red", "list", "though", "feel", "talk", "bird", "soon",
        "grown", "paper", "open", "example", "begin", "life", "always", "those", "both", "paper",
        "together", "got", "group", "often", "run", "important", "begin", "thought", "example", "children",
        "side", "feet", "car", "mile", "night", "walk", "white", "sea", "began", "grow",
        "took", "river", "four", "carry", "state", "once", "book", "hear", "stop", "without",
        "second", "later", "miss", "idea", "enough", "eat", "face", "watch", "far", "indian",
        "really", "almost", "let", "above", "girl", "sometimes", "mountain", "cut", "young", "talk",
        "soon", "list", "song", "being", "leave", "family", "it's", "on", "never", "started",
        "city", "earth", "eyes", "light", "thought", "head", "under", "story", "saw", "left",
        "don't", "few", "while", "along", "might", "close", "something", "seem", "next", "hard",
        "open", "example", "begin", "life", "always", "those", "both", "paper", "together", "got",
        "group", "often", "run", "important", "begin", "thought", "example", "children", "side", "feet",
        "car", "mile", "night", "walk", "white", "sea", "began", "grow", "took", "river",
        "four", "carry", "state", "once", "book", "hear", "stop", "without", "second", "later"
    ]

def generate_repeat_after_me_tasks():
    """Generate repeat-after-me tasks with simple dictionary words."""
    tasks = []
    dictionary = get_simple_dictionary()
    token_sizes = [10000, 50000, 100000, 250000, 450000]
    
    for tokens in token_sizes:
        # Generate random sequence of words (tokens ≈ words for simple words)
        # Use 80% of target tokens for the word sequence (leave room for prompt)
        target_words = int(tokens * 0.8)
        word_sequence = []
        for _ in range(target_words):
            word_sequence.append(random.choice(dictionary))
        
        word_string = " ".join(word_sequence)
        
        # Create prompt asking to repeat verbatim
        prompt = f"Please repeat the following text exactly as written, word for word:\n\n{word_string}\n\nRepeat the text above verbatim:"
        
        # Mark long tasks
        is_long = tokens > 50000
        
        # More accurate token estimation
        estimated_input_tokens = len(prompt) // 4  # ~4 chars per token
        
        tasks.append({
            "id": f"repeat_after_me_{tokens}_1",
            "prompt": prompt,
            "type": f"repeat_after_me_{tokens}",
            "input_tokens": estimated_input_tokens,
            "expected_output_tokens": target_words,  # Expect back the word sequence
            "is_long": is_long
        })
    
    return tasks

def generate_math_problems():
    """Generate math problems of varying difficulty."""
    problems = [
        # Simple arithmetic
        "What is 15 + 27?",
        "Calculate 144 ÷ 12",
        "What is 25% of 80?",
        
        # Intermediate
        "Solve for x: 3x + 7 = 22",
        "What is the area of a circle with radius 5?",
        "Calculate the derivative of f(x) = x³ + 2x² - 5x + 1",
        
        # Complex
        "Find the eigenvalues of the matrix [[3, 1], [1, 3]]",
        "Prove that the sum of the first n positive integers is n(n+1)/2",
        "Solve the differential equation dy/dx = xy with initial condition y(0) = 1"
    ]
    return problems

def generate_coding_problems():
    """Generate coding problems of varying complexity."""
    problems = [
        # Simple
        "Write a function that returns the sum of two numbers.",
        "Create a function to check if a number is even or odd.",
        "Write a program that prints 'Hello, World!'",
        
        # Intermediate
        "Implement a function to reverse a string without using built-in reverse methods.",
        "Write a binary search algorithm for a sorted array.",
        "Create a class for a simple calculator with basic operations.",
        
        # Complex
        "Implement a red-black tree data structure with insertion and deletion.",
        "Write a multithreaded web scraper that respects rate limits and handles errors gracefully.",
        "Design and implement a distributed hash table with consistent hashing and fault tolerance."
    ]
    return problems

def generate_story_prompts():
    """Generate creative writing prompts for different lengths."""
    prompts = [
        ("Write a haiku about winter.", 50),
        ("Write a short paragraph about a lost key.", 100),
        ("Write a one-page story about time travel.", 500),
        ("Write a detailed short story about a mysterious library.", 1000),
        ("Write a novella chapter about interstellar exploration.", 2000),
        ("Write an epic fantasy story with multiple characters and plot lines.", 5000)
    ]
    return prompts

def generate_document_tasks():
    """Generate tasks involving long documents."""
    wiki_topics = [
        "Artificial_intelligence", "Climate_change", "Quantum_computing",
        "Renaissance", "Evolution", "Nuclear_physics", "World_War_II",
        "Machine_learning", "Cryptocurrency", "Space_exploration"
    ]
    
    tasks = []
    token_sizes = [1000, 5000, 10000, 50000, 100000]
    
    for i, tokens in enumerate(token_sizes):
        topic = random.choice(wiki_topics)
        content = get_wikipedia_content(topic, tokens)
        tasks.append({
            "id": f"document_summary_{tokens}_{i+1}",
            "prompt": f"Please summarize the following text in 2-3 sentences:\n\n{content}",
            "type": f"document_summary_{tokens}",
            "input_tokens": tokens,
            "expected_output_tokens": 50
        })
    
    return tasks

def generate_repetitive_tasks():
    """Generate tasks with repetitive content to test long context."""
    tasks = []
    words = ["cat", "dog", "sun", "run", "yes"]  # Simple single-token words
    token_sizes = [1000, 10000, 50000, 100000, 250000, 500000, 900000]
    
    for i, tokens in enumerate(token_sizes):
        word = random.choice(words)
        content = generate_repetitive_content(word, tokens)
        
        # Mark long tasks
        is_long = tokens > 50000
        
        tasks.append({
            "id": f"repetitive_word_{tokens}_{i+1}",
            "prompt": f"What word is repeated in the following text?\n\n{content}",
            "type": f"repetitive_word_{tokens}",
            "input_tokens": tokens,
            "expected_output_tokens": 10,
            "is_long": is_long
        })
    
    return tasks

def generate_high_output_tasks():
    """Generate tasks that should produce very long outputs."""
    tasks = [
        {
            "id": "number_list_10k_1",
            "prompt": "List all numbers from 1 to 10000, separated by commas.",
            "type": "number_list_10k",
            "input_tokens": 20,
            "expected_output_tokens": 50000,
            "is_long": False  # Under 50K
        },
        {
            "id": "number_list_100k_1",
            "prompt": "List all numbers from 1 to 100000, separated by commas.",
            "type": "number_list_100k",
            "input_tokens": 20,
            "expected_output_tokens": 100000,
            "is_long": True  # Over 50K
        },
        {
            "id": "long_encyclopedia_1",
            "prompt": "Write a comprehensive encyclopedia entry about artificial intelligence, covering history, current state, applications, challenges, and future prospects. Be extremely detailed and thorough.",
            "type": "long_encyclopedia",
            "input_tokens": 50,
            "expected_output_tokens": 10000,
            "is_long": False  # Under 50K
        }
    ]
    return tasks

def generate_all_prompts():
    """Generate all test prompts and save to JSON files."""
    
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True)
    
    all_prompts = []
    
    # Math problems
    math_problems = generate_math_problems()
    for i, problem in enumerate(math_problems):
        all_prompts.append({
            "id": f"math_{i+1}",
            "prompt": problem,
            "type": "math",
            "input_tokens": len(problem.split()) * 1.3,  # Rough token estimate
            "expected_output_tokens": 100 if i < 3 else 300 if i < 6 else 500
        })
    
    # Coding problems
    coding_problems = generate_coding_problems()
    for i, problem in enumerate(coding_problems):
        all_prompts.append({
            "id": f"code_{i+1}",
            "prompt": problem,
            "type": "coding",
            "input_tokens": len(problem.split()) * 1.3,
            "expected_output_tokens": 200 if i < 3 else 500 if i < 6 else 1000
        })
    
    # Story prompts
    story_prompts = generate_story_prompts()
    for i, (prompt, expected_tokens) in enumerate(story_prompts):
        all_prompts.append({
            "id": f"story_{i+1}",
            "prompt": prompt,
            "type": "creative_writing",
            "input_tokens": len(prompt.split()) * 1.3,
            "expected_output_tokens": expected_tokens
        })
    
    # Document tasks
    doc_tasks = generate_document_tasks()
    all_prompts.extend(doc_tasks)
    
    # Repetitive tasks
    rep_tasks = generate_repetitive_tasks()
    all_prompts.extend(rep_tasks)
    
    # High output tasks
    high_output_tasks = generate_high_output_tasks()
    all_prompts.extend(high_output_tasks)
    
    # Repeat after me tasks
    repeat_tasks = generate_repeat_after_me_tasks()
    all_prompts.extend(repeat_tasks)
    
    # Add simple tasks
    simple_tasks = [
        {
            "id": "simple_question",
            "prompt": "What is the capital of France?",
            "type": "simple_qa",
            "input_tokens": 8,
            "expected_output_tokens": 10
        },
        {
            "id": "simple_definition",
            "prompt": "Define photosynthesis in one sentence.",
            "type": "simple_definition",
            "input_tokens": 10,
            "expected_output_tokens": 30
        }
    ]
    all_prompts.extend(simple_tasks)
    
    # Save all prompts
    with open(prompts_dir / "all_prompts.json", "w") as f:
        json.dump(all_prompts, f, indent=2)
    
    # Save prompts by category
    categories = {}
    for prompt in all_prompts:
        category = prompt["type"]
        if category not in categories:
            categories[category] = []
        categories[category].append(prompt)
    
    for category, prompts in categories.items():
        with open(prompts_dir / f"{category}_prompts.json", "w") as f:
            json.dump(prompts, f, indent=2)
    
    print(f"Generated {len(all_prompts)} prompts across {len(categories)} categories")
    print(f"Categories: {', '.join(categories.keys())}")
    
    return all_prompts

if __name__ == "__main__":
    generate_all_prompts()