# %%
"""
Advanced OpenRouter Model Analysis
Comprehensive analysis techniques and visualizations for VS Code Interactive
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# %%
# Advanced data loading with multiple file support
def load_all_performance_data():
    """Load all available performance data."""
    results_dir = Path("results")
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in results/ directory")
        return pd.DataFrame()
    
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = file.name
            all_data.append(df)
            print(f"Loaded {len(df)} rows from {file.name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Enhanced feature engineering
    combined = combined[combined['success'] == True].copy()
    combined['total_tokens'] = combined['input_tokens'] + combined['output_tokens']
    combined['time_per_input_token'] = combined['time_to_first_token'] / combined['input_tokens'].clip(lower=1)
    combined['time_per_output_token'] = (combined['total_time'] - combined['time_to_first_token']) / combined['output_tokens'].clip(lower=1)
    combined['efficiency_ratio'] = combined['throughput_tokens_per_sec'] / combined['time_to_first_token'].clip(lower=0.1)
    combined['input_output_ratio'] = combined['input_tokens'] / combined['output_tokens'].clip(lower=1)
    combined['response_latency'] = combined['time_to_first_token'] / np.log(combined['input_tokens'] + 1)
    
    return combined

df = load_all_performance_data()
if len(df) == 0:
    print("No data available for analysis")
else:
    print(f"\nCombined dataset: {len(df)} successful tests")
    print(f"Models: {len(df['model_name'].unique())}")
    print(f"Task types: {len(df['prompt_type'].unique())}")

# %%
# Statistical significance testing between models
if len(df) > 0 and len(df['model_name'].unique()) > 1:
    print("=== STATISTICAL SIGNIFICANCE TESTS ===")
    
    models = df['model_name'].unique()
    metrics = ['throughput_tokens_per_sec', 'time_to_first_token', 'efficiency_ratio']
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        model_groups = [df[df['model_name'] == model][metric].values for model in models]
        
        # Remove empty groups
        model_groups = [group for group in model_groups if len(group) > 0]
        model_names = [model for model in models if len(df[df['model_name'] == model]) > 0]
        
        if len(model_groups) >= 2:
            # ANOVA test
            f_stat, p_value = stats.f_oneway(*model_groups)
            print(f"  ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.6f}")
            
            if p_value < 0.05:
                print("  *** Significant difference between models ***")
                
                # Pairwise comparisons
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        t_stat, p_val = stats.ttest_ind(model_groups[i], model_groups[j])
                        print(f"    {model_names[i][:20]...} vs {model_names[j][:20]...}: p={p_val:.4f}")
            else:
                print("  No significant difference between models")

# %%
# Advanced correlation heatmap
if len(df) > 0:
    print("=== CORRELATION ANALYSIS ===")
    
    # Select numeric columns for correlation
    numeric_cols = ['input_tokens', 'output_tokens', 'total_tokens', 
                   'time_to_first_token', 'total_time', 'throughput_tokens_per_sec',
                   'time_per_input_token', 'time_per_output_token', 'efficiency_ratio',
                   'input_output_ratio', 'response_latency']
    
    # Filter to existing columns
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) > 2:
        corr_matrix = df[available_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Performance Metrics Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Highlight strong correlations
        strong_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append((corr_matrix.index[i], corr_matrix.columns[j], corr_val))
        
        if strong_corr:
            print("\nStrong correlations (|r| > 0.7):")
            for var1, var2, corr in strong_corr:
                print(f"  {var1} ↔ {var2}: {corr:.3f}")

# %%
# Performance clustering analysis
if len(df) > 0 and len(df['model_name'].unique()) > 1:
    print("=== PERFORMANCE CLUSTERING ===")
    
    # Prepare data for clustering
    feature_cols = ['throughput_tokens_per_sec', 'time_to_first_token', 'efficiency_ratio']
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) >= 2:
        # Aggregate by model
        model_features = df.groupby('model_name')[available_features].mean()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(model_features)
        
        # PCA for visualization
        if len(available_features) > 2:
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(scaled_features)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], s=100, alpha=0.7)
            
            # Add model labels
            for i, model in enumerate(model_features.index):
                plt.annotate(model.split('/')[-1], (pca_features[i, 0], pca_features[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('Model Performance Clustering (PCA)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print(f"PCA explains {pca.explained_variance_ratio_.sum():.1%} of variance")

# %%
# Task complexity analysis
if len(df) > 0 and len(df['prompt_type'].unique()) > 1:
    print("=== TASK COMPLEXITY ANALYSIS ===")
    
    # Define complexity score based on token counts and performance
    df['complexity_score'] = (
        np.log(df['input_tokens'] + 1) * 0.3 +
        np.log(df['expected_output_tokens'] + 1) * 0.4 +
        (1 / (df['throughput_tokens_per_sec'] + 1)) * 1000 * 0.3
    )
    
    task_complexity = df.groupby('prompt_type').agg({
        'complexity_score': 'mean',
        'input_tokens': 'mean',
        'output_tokens': 'mean',
        'throughput_tokens_per_sec': 'mean',
        'time_to_first_token': 'mean'
    }).round(2)
    
    task_complexity = task_complexity.sort_values('complexity_score', ascending=False)
    print("Task complexity ranking:")
    print(task_complexity)
    
    # Visualize task complexity vs performance
    plt.figure(figsize=(12, 8))
    
    task_stats = df.groupby('prompt_type').agg({
        'complexity_score': 'mean',
        'throughput_tokens_per_sec': 'mean'
    })
    
    plt.scatter(task_stats['complexity_score'], task_stats['throughput_tokens_per_sec'], 
               s=100, alpha=0.7)
    
    for task, row in task_stats.iterrows():
        plt.annotate(task[:15] + ('...' if len(task) > 15 else ''), 
                    (row['complexity_score'], row['throughput_tokens_per_sec']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Task Complexity Score')
    plt.ylabel('Average Throughput (tokens/sec)')
    plt.title('Task Complexity vs Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
# Performance scaling analysis
if len(df) > 0:
    print("=== PERFORMANCE SCALING ANALYSIS ===")
    
    # Bin by input size for scaling analysis
    df['input_size_bin'] = pd.cut(df['input_tokens'], bins=[0, 10, 50, 500, 5000, 50000, np.inf],
                                 labels=['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge'])
    
    scaling_analysis = df.groupby(['model_name', 'input_size_bin']).agg({
        'throughput_tokens_per_sec': ['mean', 'count'],
        'time_to_first_token': 'mean',
        'input_tokens': 'mean'
    }).round(2)
    
    print("Throughput scaling by input size:")
    for model in df['model_name'].unique():
        print(f"\n{model}:")
        model_scaling = scaling_analysis.loc[model] if model in scaling_analysis.index else None
        if model_scaling is not None:
            print(model_scaling)
    
    # Plot scaling curves
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for model in df['model_name'].unique()[:3]:  # Limit to first 3 models for clarity
        model_data = df[df['model_name'] == model]
        if len(model_data) > 5:
            # Create input size bins for this model
            bins = np.logspace(np.log10(model_data['input_tokens'].min()), 
                             np.log10(model_data['input_tokens'].max()), 6)
            model_data['log_input_bin'] = pd.cut(model_data['input_tokens'], bins=bins)
            
            scaling_data = model_data.groupby('log_input_bin').agg({
                'throughput_tokens_per_sec': 'mean',
                'input_tokens': 'mean'
            }).dropna()
            
            if len(scaling_data) > 1:
                plt.plot(scaling_data['input_tokens'], scaling_data['throughput_tokens_per_sec'], 
                        'o-', label=model.split('/')[-1], alpha=0.8)
    
    plt.xlabel('Input Tokens (log scale)')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Throughput Scaling with Input Size')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for model in df['model_name'].unique()[:3]:
        model_data = df[df['model_name'] == model]
        if len(model_data) > 5:
            bins = np.logspace(np.log10(model_data['input_tokens'].min()), 
                             np.log10(model_data['input_tokens'].max()), 6)
            model_data['log_input_bin'] = pd.cut(model_data['input_tokens'], bins=bins)
            
            scaling_data = model_data.groupby('log_input_bin').agg({
                'time_to_first_token': 'mean',
                'input_tokens': 'mean'
            }).dropna()
            
            if len(scaling_data) > 1:
                plt.plot(scaling_data['input_tokens'], scaling_data['time_to_first_token'], 
                        'o-', label=model.split('/')[-1], alpha=0.8)
    
    plt.xlabel('Input Tokens (log scale)')
    plt.ylabel('Time to First Token (s)')
    plt.title('Response Time Scaling')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
# Export comprehensive analysis
if len(df) > 0:
    print("=== EXPORTING COMPREHENSIVE ANALYSIS ===")
    
    # Create analysis directory
    Path("analysis").mkdir(exist_ok=True)
    
    # Save enhanced dataset
    df.to_csv('analysis/comprehensive_analysis.csv', index=False)
    
    # Generate model comparison report
    models = df['model_name'].unique()
    comparison_report = {}
    
    for model in models:
        model_data = df[df['model_name'] == model]
        comparison_report[model] = {
            'test_count': len(model_data),
            'avg_throughput': model_data['throughput_tokens_per_sec'].mean(),
            'throughput_std': model_data['throughput_tokens_per_sec'].std(),
            'avg_ttft': model_data['time_to_first_token'].mean(),
            'ttft_std': model_data['time_to_first_token'].std(),
            'efficiency_ratio': model_data['efficiency_ratio'].mean(),
            'best_task_type': model_data.groupby('prompt_type')['throughput_tokens_per_sec'].mean().idxmax(),
            'worst_task_type': model_data.groupby('prompt_type')['throughput_tokens_per_sec'].mean().idxmin()
        }
    
    # Save comparison
    import json
    with open('analysis/model_comparison.json', 'w') as f:
        json.dump(comparison_report, f, indent=2, default=str)
    
    print("Comprehensive analysis saved to analysis/ directory")
    print("Files created:")
    print("- comprehensive_analysis.csv")
    print("- model_comparison.json")

print("\n🚀 Advanced analysis complete!")
print("\nNext steps:")
print("- Examine the correlation heatmap for insights")
print("- Check statistical significance results")
print("- Review task complexity rankings")  
print("- Analyze performance scaling patterns")
print("- Use the exported data for custom analysis")

# %%