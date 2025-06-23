# %%
"""
Interactive Analysis for OpenRouter Model Performance
Run this file in VS Code with Python Interactive to see graphs inline
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')

# %%
# Configuration - Edit these paths as needed
CSV_FILES = [
    "results/performance_results.csv",
    "results/family_test_results.csv",
    # Add more CSV files here
]

# Ensure we're in the right directory
import os
if 'measuring-performance-of-OR-models' not in os.getcwd():
    print("Warning: Make sure you're in the measuring-performance-of-OR-models directory")

# %%
# Load and combine data
def load_performance_data(csv_files: List[str]) -> pd.DataFrame:
    """Load data from CSV files."""
    dfs = []
    for file in csv_files:
        if Path(file).exists():
            df = pd.read_csv(file)
            print(f"Loaded {len(df)} rows from {file}")
            dfs.append(df)
        else:
            print(f"Warning: File {file} not found")
    
    if not dfs:
        raise ValueError("No valid CSV files found")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Filter successful tests only
    successful_df = combined_df[combined_df['success'] == True].copy()
    
    # Calculate additional metrics
    successful_df['total_tokens'] = successful_df['input_tokens'] + successful_df['output_tokens']
    successful_df['time_per_input_token'] = successful_df['time_to_first_token'] / successful_df['input_tokens'].clip(lower=1)
    successful_df['time_per_total_token'] = successful_df['total_time'] / successful_df['total_tokens'].clip(lower=1)
    
    return successful_df

df = load_performance_data(CSV_FILES)
print(f"\nTotal successful tests: {len(df)}")
print(f"Models tested: {df['model_name'].unique().tolist()}")
print(f"Task types: {df['prompt_type'].unique().tolist()}")

# %%
# Data overview
print("=== DATA OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nBasic statistics:")
df[['input_tokens', 'output_tokens', 'time_to_first_token', 'throughput_tokens_per_sec', 'total_time']].describe()

# %%
# Model performance summary
print("=== MODEL PERFORMANCE SUMMARY ===")
models = df['model_name'].unique()

for model in models:
    model_data = df[df['model_name'] == model]
    print(f"\n{model}:")
    print(f"  Tests: {len(model_data)}")
    print(f"  Avg throughput: {model_data['throughput_tokens_per_sec'].mean():.1f} tokens/sec")
    print(f"  Avg TTFT: {model_data['time_to_first_token'].mean():.3f}s")
    print(f"  Avg input tokens: {model_data['input_tokens'].mean():.0f}")
    print(f"  Avg output tokens: {model_data['output_tokens'].mean():.0f}")

# %%
# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

models = df['model_name'].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(models)))

# %%
# Visualization 1: Time to First Token vs Input Tokens
plt.figure(figsize=(12, 8))
for i, model in enumerate(models):
    model_data = df[df['model_name'] == model]
    plt.scatter(model_data['input_tokens'], model_data['time_to_first_token'], 
              alpha=0.7, label=model, color=colors[i], s=50)

plt.xlabel('Input Tokens')
plt.ylabel('Time to First Token (s)')
plt.title('Time to First Token vs Input Tokens')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Visualization 2: Throughput vs Output Tokens
plt.figure(figsize=(12, 8))
for i, model in enumerate(models):
    model_data = df[df['model_name'] == model]
    plt.scatter(model_data['output_tokens'], model_data['throughput_tokens_per_sec'], 
              alpha=0.7, label=model, color=colors[i], s=50)

plt.xlabel('Output Tokens')
plt.ylabel('Throughput (tokens/sec)')
plt.title('Throughput vs Output Tokens')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Visualization 3: Total Time vs Total Tokens
plt.figure(figsize=(12, 8))
for i, model in enumerate(models):
    model_data = df[df['model_name'] == model]
    plt.scatter(model_data['total_tokens'], model_data['total_time'], 
              alpha=0.7, label=model, color=colors[i], s=50)

plt.xlabel('Total Tokens (Input + Output)')
plt.ylabel('Total Time (s)')
plt.title('Total Time vs Total Tokens')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Visualization 4: Throughput Distribution
if len(models) > 1:
    plt.figure(figsize=(12, 8))
    throughput_data = []
    labels = []
    
    for model in models:
        model_data = df[df['model_name'] == model]
        throughput_data.append(model_data['throughput_tokens_per_sec'].values)
        labels.append(model.split('/')[-1])  # Use shorter names
    
    plt.boxplot(throughput_data, labels=labels)
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Throughput Distribution by Model')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    # Single model histogram
    plt.figure(figsize=(12, 6))
    plt.hist(df['throughput_tokens_per_sec'], bins=20, alpha=0.7, color=colors[0])
    plt.xlabel('Throughput (tokens/sec)')
    plt.ylabel('Frequency')
    plt.title(f'Throughput Distribution - {models[0]}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
# Visualization 5: Performance by Task Type (if multiple task types exist)
if len(df['prompt_type'].unique()) > 1:
    plt.figure(figsize=(15, 10))
    
    # Group by prompt type and calculate average throughput
    type_performance = df.groupby(['prompt_type', 'model_name'])['throughput_tokens_per_sec'].mean().reset_index()
    
    # Create a pivot table for better visualization
    pivot_data = type_performance.pivot(index='prompt_type', columns='model_name', values='throughput_tokens_per_sec')
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Throughput (tokens/sec)'})
    plt.title('Average Throughput by Task Type and Model')
    plt.xlabel('Model')
    plt.ylabel('Task Type')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# %%
# Visualization 6: Time to First Token Distribution
if len(models) > 1:
    plt.figure(figsize=(12, 8))
    ttft_data = []
    labels = []
    
    for model in models:
        model_data = df[df['model_name'] == model]
        ttft_data.append(model_data['time_to_first_token'].values)
        labels.append(model.split('/')[-1])
    
    plt.boxplot(ttft_data, labels=labels)
    plt.ylabel('Time to First Token (s)')
    plt.title('Time to First Token Distribution by Model')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    # Single model histogram
    plt.figure(figsize=(12, 6))
    plt.hist(df['time_to_first_token'], bins=20, alpha=0.7, color=colors[0])
    plt.xlabel('Time to First Token (s)')
    plt.ylabel('Frequency')
    plt.title(f'Time to First Token Distribution - {models[0]}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
# Calculate detailed statistics
def calculate_model_stats(df: pd.DataFrame, model: str) -> Dict[str, Any]:
    """Calculate detailed statistics for a model."""
    model_data = df[df['model_name'] == model]
    
    if len(model_data) == 0:
        return {}
    
    stats = {
        'total_tests': len(model_data),
        'avg_time_to_first_token': model_data['time_to_first_token'].mean(),
        'std_time_to_first_token': model_data['time_to_first_token'].std(),
        'avg_time_per_input_token': model_data['time_per_input_token'].mean(),
        'avg_throughput': model_data['throughput_tokens_per_sec'].mean(),
        'std_throughput': model_data['throughput_tokens_per_sec'].std(),
        'avg_total_time': model_data['total_time'].mean(),
        'avg_time_per_total_token': model_data['time_per_total_token'].mean(),
        'median_throughput': model_data['throughput_tokens_per_sec'].median(),
        'min_throughput': model_data['throughput_tokens_per_sec'].min(),
        'max_throughput': model_data['throughput_tokens_per_sec'].max(),
        'avg_input_tokens': model_data['input_tokens'].mean(),
        'avg_output_tokens': model_data['output_tokens'].mean(),
    }
    
    # Calculate correlations
    correlations = {}
    for metric in ['time_to_first_token', 'throughput_tokens_per_sec', 'total_time']:
        correlations[f'{metric}_vs_input_tokens'] = model_data['input_tokens'].corr(model_data[metric])
        correlations[f'{metric}_vs_output_tokens'] = model_data['output_tokens'].corr(model_data[metric])
        correlations[f'{metric}_vs_total_tokens'] = model_data['total_tokens'].corr(model_data[metric])
    
    stats['correlations'] = correlations
    return stats

# Calculate stats for all models
print("=== DETAILED STATISTICS ===")
all_stats = {}
for model in models:
    stats = calculate_model_stats(df, model)
    all_stats[model] = stats
    
    print(f"\n{model}:")
    print(f"  Total tests: {stats['total_tests']}")
    print(f"  Avg throughput: {stats['avg_throughput']:.2f} ± {stats['std_throughput']:.2f} tokens/sec")
    print(f"  Median throughput: {stats['median_throughput']:.2f} tokens/sec")
    print(f"  Throughput range: {stats['min_throughput']:.2f} - {stats['max_throughput']:.2f} tokens/sec")
    print(f"  Avg TTFT: {stats['avg_time_to_first_token']:.3f} ± {stats['std_time_to_first_token']:.3f}s")
    print(f"  Avg time per input token: {stats['avg_time_per_input_token']:.4f}s")

# %%
# Correlation analysis
print("=== CORRELATION ANALYSIS ===")
for model in models:
    if model in all_stats:
        print(f"\n{model}:")
        corr = all_stats[model]['correlations']
        print(f"  TTFT vs input tokens: {corr['time_to_first_token_vs_input_tokens']:.3f}")
        print(f"  TTFT vs output tokens: {corr['time_to_first_token_vs_output_tokens']:.3f}")
        print(f"  Throughput vs input tokens: {corr['throughput_tokens_per_sec_vs_input_tokens']:.3f}")
        print(f"  Throughput vs output tokens: {corr['throughput_tokens_per_sec_vs_output_tokens']:.3f}")
        print(f"  Total time vs total tokens: {corr['total_time_vs_total_tokens']:.3f}")

# %%
# Custom analysis section - Add your own explorations here!
print("=== CUSTOM ANALYSIS SECTION ===")
print("Edit this section to add your own analysis...")

# Example: Look at task-specific performance
if len(df['prompt_type'].unique()) > 1:
    print("\nTask-specific performance:")
    task_perf = df.groupby('prompt_type').agg({
        'throughput_tokens_per_sec': ['mean', 'std', 'count'],
        'time_to_first_token': ['mean', 'std'],
        'input_tokens': 'mean',
        'output_tokens': 'mean'
    }).round(2)
    print(task_perf)

# %%
# Export results for further analysis
print("=== EXPORTING RESULTS ===")

# Save processed dataframe
df.to_csv('analysis/processed_results.csv', index=False)
print("Saved processed results to: analysis/processed_results.csv")

# Save statistics
with open('analysis/interactive_statistics.json', 'w') as f:
    json.dump(all_stats, f, indent=2)
print("Saved statistics to: analysis/interactive_statistics.json")

print("\nAnalysis complete! 🎉")
print("You can now:")
print("- Modify the custom analysis section above")
print("- Add new visualizations")  
print("- Explore specific models or task types")
print("- Export data for other tools")

# %%