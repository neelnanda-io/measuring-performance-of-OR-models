#!/usr/bin/env python3
"""
Analyze performance measurement results and generate reports.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')

class ResultsAnalyzer:
    def __init__(self, csv_files: List[str]):
        self.csv_files = csv_files
        self.df = self.load_data()
        self.models = self.df['model_name'].unique().tolist()
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV files."""
        dfs = []
        for file in self.csv_files:
            if Path(file).exists():
                df = pd.read_csv(file)
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
    
    def calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics for each model."""
        summary = {}
        
        for model in self.models:
            model_data = self.df[self.df['model_name'] == model]
            
            if len(model_data) == 0:
                continue
                
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
            summary[model] = stats
        
        return summary
    
    def create_visualizations(self) -> None:
        """Create various visualizations."""
        analysis_dir = Path("analysis")
        analysis_dir.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.models)))
        
        # 1. Time to first token vs Input tokens
        plt.figure(figsize=(12, 8))
        for i, model in enumerate(self.models):
            model_data = self.df[self.df['model_name'] == model]
            plt.scatter(model_data['input_tokens'], model_data['time_to_first_token'], 
                      alpha=0.6, label=model, color=colors[i])
        
        plt.xlabel('Input Tokens')
        plt.ylabel('Time to First Token (s)')
        plt.title('Time to First Token vs Input Tokens')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(analysis_dir / 'time_to_first_token_vs_input.png', dpi=300)
        plt.close()
        
        # 2. Throughput vs Output tokens
        plt.figure(figsize=(12, 8))
        for i, model in enumerate(self.models):
            model_data = self.df[self.df['model_name'] == model]
            plt.scatter(model_data['output_tokens'], model_data['throughput_tokens_per_sec'], 
                      alpha=0.6, label=model, color=colors[i])
        
        plt.xlabel('Output Tokens')
        plt.ylabel('Throughput (tokens/sec)')
        plt.title('Throughput vs Output Tokens')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(analysis_dir / 'throughput_vs_output.png', dpi=300)
        plt.close()
        
        # 3. Total time vs Total tokens
        plt.figure(figsize=(12, 8))
        for i, model in enumerate(self.models):
            model_data = self.df[self.df['model_name'] == model]
            plt.scatter(model_data['total_tokens'], model_data['total_time'], 
                      alpha=0.6, label=model, color=colors[i])
        
        plt.xlabel('Total Tokens (Input + Output)')
        plt.ylabel('Total Time (s)')
        plt.title('Total Time vs Total Tokens')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(analysis_dir / 'total_time_vs_total_tokens.png', dpi=300)
        plt.close()
        
        # 4. Throughput distribution
        plt.figure(figsize=(12, 8))
        throughput_data = []
        labels = []
        
        for model in self.models:
            model_data = self.df[self.df['model_name'] == model]
            throughput_data.append(model_data['throughput_tokens_per_sec'].values)
            labels.append(model)
        
        plt.boxplot(throughput_data, labels=labels)
        plt.ylabel('Throughput (tokens/sec)')
        plt.title('Throughput Distribution by Model')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(analysis_dir / 'throughput_distribution.png', dpi=300)
        plt.close()
        
        # 5. Performance by task type
        if len(self.df['prompt_type'].unique()) > 1:
            plt.figure(figsize=(15, 10))
            
            # Group by prompt type and calculate average throughput
            type_performance = self.df.groupby(['prompt_type', 'model_name'])['throughput_tokens_per_sec'].mean().reset_index()
            
            # Create a pivot table for better visualization
            pivot_data = type_performance.pivot(index='prompt_type', columns='model_name', values='throughput_tokens_per_sec')
            
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Throughput (tokens/sec)'})
            plt.title('Average Throughput by Task Type and Model')
            plt.xlabel('Model')
            plt.ylabel('Task Type')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(analysis_dir / 'performance_by_task_type.png', dpi=300)
            plt.close()
        
        # 6. Time to first token distribution
        plt.figure(figsize=(12, 8))
        ttft_data = []
        labels = []
        
        for model in self.models:
            model_data = self.df[self.df['model_name'] == model]
            ttft_data.append(model_data['time_to_first_token'].values)
            labels.append(model)
        
        plt.boxplot(ttft_data, labels=labels)
        plt.ylabel('Time to First Token (s)')
        plt.title('Time to First Token Distribution by Model')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(analysis_dir / 'ttft_distribution.png', dpi=300)
        plt.close()
        
        print(f"Visualizations saved to {analysis_dir}")
    
    def generate_report(self, summary_stats: Dict[str, Any]) -> str:
        """Generate a comprehensive report."""
        report = []
        
        # Executive Summary
        report.append("# OpenRouter Model Performance Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append("## Executive Summary")
        report.append("")
        
        if len(self.models) == 1:
            model = self.models[0]
            stats = summary_stats[model]
            report.append(f"**Model Tested:** {model}")
            report.append(f"**Total Tests:** {stats['total_tests']}")
            report.append(f"**Average Throughput:** {stats['avg_throughput']:.2f} tokens/sec")
            report.append(f"**Average Time to First Token:** {stats['avg_time_to_first_token']:.3f} seconds")
            report.append(f"**Median Throughput:** {stats['median_throughput']:.2f} tokens/sec")
            report.append("")
        else:
            report.append("**Models Compared:**")
            for model in self.models:
                stats = summary_stats[model]
                report.append(f"- {model}: {stats['avg_throughput']:.2f} tokens/sec avg throughput")
            report.append("")
        
        # Detailed Results
        report.append("## Detailed Performance Metrics")
        report.append("")
        
        for model in self.models:
            stats = summary_stats[model]
            report.append(f"### {model}")
            report.append("")
            report.append(f"**Basic Metrics:**")
            report.append(f"- Total tests completed: {stats['total_tests']}")
            report.append(f"- Average input tokens: {stats['avg_input_tokens']:.0f}")
            report.append(f"- Average output tokens: {stats['avg_output_tokens']:.0f}")
            report.append("")
            report.append(f"**Speed Metrics:**")
            report.append(f"- Average time to first token: {stats['avg_time_to_first_token']:.3f}s (±{stats['std_time_to_first_token']:.3f}s)")
            report.append(f"- Average time per input token: {stats['avg_time_per_input_token']:.4f}s")
            report.append(f"- Average throughput: {stats['avg_throughput']:.2f} tokens/sec (±{stats['std_throughput']:.2f})")
            report.append(f"- Median throughput: {stats['median_throughput']:.2f} tokens/sec")
            report.append(f"- Throughput range: {stats['min_throughput']:.2f} - {stats['max_throughput']:.2f} tokens/sec")
            report.append(f"- Average total time: {stats['avg_total_time']:.2f}s")
            report.append(f"- Average time per total token: {stats['avg_time_per_total_token']:.4f}s")
            report.append("")
            report.append(f"**Correlations:**")
            corr = stats['correlations']
            report.append(f"- Time to first token correlation with input tokens: {corr['time_to_first_token_vs_input_tokens']:.3f}")
            report.append(f"- Throughput correlation with output tokens: {corr['throughput_tokens_per_sec_vs_output_tokens']:.3f}")
            report.append(f"- Total time correlation with total tokens: {corr['total_time_vs_total_tokens']:.3f}")
            report.append("")
        
        # Key Findings
        report.append("## Key Findings")
        report.append("")
        
        if len(self.models) == 1:
            model = self.models[0]
            stats = summary_stats[model]
            
            # Analysis of single model
            report.append("**Performance Characteristics:**")
            if stats['avg_throughput'] > 50:
                report.append("- High throughput model - excellent for bulk text generation")
            elif stats['avg_throughput'] > 20:
                report.append("- Moderate throughput model - good for general use")
            else:
                report.append("- Lower throughput model - may be optimized for other factors")
            
            if stats['avg_time_to_first_token'] < 1.0:
                report.append("- Fast response time - good for interactive applications")
            elif stats['avg_time_to_first_token'] < 3.0:
                report.append("- Moderate response time - suitable for most applications")
            else:
                report.append("- Slower response time - may require optimization for real-time use")
            
            # Correlation insights
            ttft_corr = stats['correlations']['time_to_first_token_vs_input_tokens']
            if ttft_corr > 0.5:
                report.append("- Time to first token scales significantly with input length")
            elif ttft_corr > 0.2:
                report.append("- Time to first token shows moderate scaling with input length")
            else:
                report.append("- Time to first token is relatively consistent across input lengths")
            
        else:
            # Compare multiple models
            best_throughput = max(summary_stats.keys(), key=lambda x: summary_stats[x]['avg_throughput'])
            best_ttft = min(summary_stats.keys(), key=lambda x: summary_stats[x]['avg_time_to_first_token'])
            
            report.append("**Model Comparison:**")
            report.append(f"- **Highest throughput:** {best_throughput} ({summary_stats[best_throughput]['avg_throughput']:.2f} tokens/sec)")
            report.append(f"- **Fastest response:** {best_ttft} ({summary_stats[best_ttft]['avg_time_to_first_token']:.3f}s)")
        
        # Recommendations
        report.append("")
        report.append("## Recommendations")
        report.append("")
        
        if len(self.models) == 1:
            model = self.models[0]
            stats = summary_stats[model]
            
            if stats['avg_throughput'] > 30 and stats['avg_time_to_first_token'] < 2.0:
                report.append("- This model shows excellent performance characteristics for most applications")
            elif stats['avg_throughput'] < 10:
                report.append("- Consider testing with shorter prompts or different models for better throughput")
            
            if stats['std_throughput'] / stats['avg_throughput'] > 0.5:
                report.append("- High throughput variability - performance may depend significantly on task type")
        
        else:
            report.append("- Choose models based on your specific use case:")
            for model in self.models:
                stats = summary_stats[model]
                if stats['avg_throughput'] > 30:
                    report.append(f"  - {model}: Best for high-volume text generation")
                elif stats['avg_time_to_first_token'] < 1.0:
                    report.append(f"  - {model}: Best for interactive applications")
        
        return "\n".join(report)
    
    def save_summary_stats(self, summary_stats: Dict[str, Any]) -> None:
        """Save summary statistics to JSON."""
        analysis_dir = Path("analysis")
        analysis_dir.mkdir(exist_ok=True)
        
        with open(analysis_dir / "summary_statistics.json", "w") as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Summary statistics saved to {analysis_dir / 'summary_statistics.json'}")
    
    def run_analysis(self) -> None:
        """Run the complete analysis."""
        print(f"Analyzing {len(self.df)} successful test results across {len(self.models)} model(s)")
        
        # Calculate summary statistics
        summary_stats = self.calculate_summary_stats()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        report = self.generate_report(summary_stats)
        
        # Save everything
        analysis_dir = Path("analysis")
        analysis_dir.mkdir(exist_ok=True)
        
        with open(analysis_dir / "performance_report.md", "w") as f:
            f.write(report)
        
        self.save_summary_stats(summary_stats)
        
        print(f"Analysis complete! Report saved to {analysis_dir / 'performance_report.md'}")
        print("\nQuick Summary:")
        for model in self.models:
            stats = summary_stats[model]
            print(f"- {model}: {stats['avg_throughput']:.1f} tokens/sec, {stats['avg_time_to_first_token']:.3f}s TTFT")

def main():
    parser = argparse.ArgumentParser(description="Analyze OpenRouter performance results")
    parser.add_argument("csv_files", nargs="+", help="CSV files containing performance results")
    parser.add_argument("--output-dir", default="analysis", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ResultsAnalyzer(args.csv_files)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()