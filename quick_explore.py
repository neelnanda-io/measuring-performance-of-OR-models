# %%
"""
Quick Model Performance Explorer
Perfect for VS Code Interactive mode - lightweight and fast exploration
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# %%
# Quick setup - just load the most recent results
latest_results = "results/performance_results.csv"
if Path(latest_results).exists():
    df = pd.read_csv(latest_results)
    df = df[df['success'] == True]  # Only successful tests
    print(f"Loaded {len(df)} successful tests")
    print(f"Models: {df['model_name'].unique()}")
else:
    print("No results found - run some tests first!")

# %%
# Quick stats overview
if 'df' in locals():
    print("=== QUICK STATS ===")
    stats = df.groupby('model_name').agg({
        'throughput_tokens_per_sec': ['mean', 'median', 'std'],
        'time_to_first_token': ['mean', 'median', 'std'],
        'total_time': 'mean',
        'input_tokens': 'mean',
        'output_tokens': 'mean'
    }).round(2)
    
    print(stats)

# %%
# Quick throughput plot
if 'df' in locals():
    plt.figure(figsize=(10, 6))
    
    models = df['model_name'].unique()
    if len(models) == 1:
        # Single model - show distribution
        plt.hist(df['throughput_tokens_per_sec'], bins=15, alpha=0.7, edgecolor='black')
        plt.title(f'Throughput Distribution - {models[0].split("/")[-1]}')
        plt.xlabel('Throughput (tokens/sec)')
        plt.ylabel('Count')
    else:
        # Multiple models - compare
        for model in models:
            model_data = df[df['model_name'] == model]
            plt.scatter(model_data['output_tokens'], model_data['throughput_tokens_per_sec'], 
                       label=model.split('/')[-1], alpha=0.7, s=50)
        plt.xlabel('Output Tokens')
        plt.ylabel('Throughput (tokens/sec)')
        plt.title('Throughput vs Output Size')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
# Quick TTFT analysis
if 'df' in locals():
    plt.figure(figsize=(10, 6))
    
    models = df['model_name'].unique()
    if len(models) == 1:
        # Single model - TTFT vs input size
        plt.scatter(df['input_tokens'], df['time_to_first_token'], alpha=0.7, s=50)
        plt.xlabel('Input Tokens')
        plt.ylabel('Time to First Token (s)')
        plt.title(f'Response Time Scaling - {models[0].split("/")[-1]}')
        
        # Add trend line
        z = np.polyfit(df['input_tokens'], df['time_to_first_token'], 1)
        p = np.poly1d(z)
        plt.plot(df['input_tokens'].sort_values(), p(df['input_tokens'].sort_values()), 
                "r--", alpha=0.8, linewidth=2)
    else:
        # Multiple models - compare TTFT
        for model in models:
            model_data = df[df['model_name'] == model]
            plt.scatter(model_data['input_tokens'], model_data['time_to_first_token'], 
                       label=model.split('/')[-1], alpha=0.7, s=50)
        plt.xlabel('Input Tokens')
        plt.ylabel('Time to First Token (s)')
        plt.title('Response Time Comparison')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
# Task type performance (if available)
if 'df' in locals() and len(df['prompt_type'].unique()) > 1:
    print("=== PERFORMANCE BY TASK TYPE ===")
    
    task_stats = df.groupby('prompt_type').agg({
        'throughput_tokens_per_sec': 'mean',
        'time_to_first_token': 'mean',
        'input_tokens': 'mean',
        'output_tokens': 'mean'
    }).round(1)
    
    print(task_stats.sort_values('throughput_tokens_per_sec', ascending=False))
    
    # Plot task performance
    plt.figure(figsize=(12, 6))
    task_throughput = df.groupby('prompt_type')['throughput_tokens_per_sec'].mean().sort_values(ascending=True)
    
    plt.barh(range(len(task_throughput)), task_throughput.values)
    plt.yticks(range(len(task_throughput)), task_throughput.index)
    plt.xlabel('Average Throughput (tokens/sec)')
    plt.title('Performance by Task Type')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# %%
# Model efficiency analysis
if 'df' in locals():
    print("=== EFFICIENCY METRICS ===")
    
    # Calculate efficiency metrics
    df['efficiency_score'] = df['throughput_tokens_per_sec'] / df['time_to_first_token']
    df['tokens_per_sec_total'] = df['total_tokens'] / df['total_time']
    
    efficiency = df.groupby('model_name').agg({
        'efficiency_score': 'mean',
        'tokens_per_sec_total': 'mean',
        'throughput_tokens_per_sec': 'mean',
        'time_to_first_token': 'mean'
    }).round(2)
    
    print("Efficiency Score = Throughput / Time to First Token")
    print(efficiency.sort_values('efficiency_score', ascending=False))

# %%
# Quick model comparison (if multiple models)
if 'df' in locals() and len(df['model_name'].unique()) > 1:
    print("=== MODEL COMPARISON ===")
    
    comparison = df.groupby('model_name').agg({
        'throughput_tokens_per_sec': ['mean', 'median'],
        'time_to_first_token': ['mean', 'median'],
        'total_time': 'mean'
    }).round(2)
    
    print("Throughput comparison:")
    throughput_comparison = comparison['throughput_tokens_per_sec'].sort_values('mean', ascending=False)
    print(throughput_comparison)
    
    print("\nResponse time comparison:")
    ttft_comparison = comparison['time_to_first_token'].sort_values('mean', ascending=True)
    print(ttft_comparison)

# %%
# Data export for further analysis
if 'df' in locals():
    print("=== EXPORT OPTIONS ===")
    
    # Save enhanced dataframe
    df['efficiency_score'] = df.get('efficiency_score', df['throughput_tokens_per_sec'] / df['time_to_first_token'])
    df['tokens_per_sec_total'] = df.get('tokens_per_sec_total', (df['input_tokens'] + df['output_tokens']) / df['total_time'])
    
    export_path = 'analysis/enhanced_results.csv'
    df.to_csv(export_path, index=False)
    print(f"Enhanced results saved to: {export_path}")
    
    # Summary for quick reference
    summary = df.groupby('model_name').agg({
        'throughput_tokens_per_sec': ['mean', 'std'],
        'time_to_first_token': ['mean', 'std'],
        'efficiency_score': 'mean'
    }).round(2)
    
    print("\nQuick summary:")
    print(summary)

print("\n🔬 Quick exploration complete!")
print("Tips:")
print("- Edit any cell above and re-run to explore different aspects")
print("- Add new cells below for custom analysis")
print("- Use df.query() to filter specific conditions")
print("- Try df.groupby() for custom aggregations")

# %%