# OpenRouter Model Performance Measurement

A comprehensive benchmarking tool for measuring the performance characteristics of OpenRouter language models across various task types and input/output sizes.

## Features

- **Comprehensive Testing**: 70+ diverse prompts across 17+ categories
- **Multi-Modal Support**: Text, code, math, vision tasks, and long context
- **Detailed Metrics**: Time to first token, throughput, total response time
- **Batch Testing**: Test multiple models automatically
- **Rich Analysis**: Statistical summaries, correlations, and visualizations
- **Professional Reports**: Automated markdown reports with key insights

## Test Categories

- **Simple Tasks**: Basic Q&A, definitions
- **Mathematical Problems**: From arithmetic to complex proofs
- **Coding Challenges**: Simple functions to complex algorithms
- **Creative Writing**: Haikus to full novellas
- **Document Processing**: Summarization of 1K-100K token documents
- **Long Context**: Repetitive text recognition up to 100K tokens
- **High Output**: Number sequences, encyclopedia entries
- **Vision Tasks**: Geometric shape recognition (1-5 images per prompt)

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt` 
3. Add your OpenRouter API key to `api.secrets` file:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```
4. Generate test data:
   ```bash
   python generate_prompts.py
   python generate_images.py
   ```

## Usage

### Single Model Testing
```bash
# Test specific model
python measure_performance.py google/gemini-2.0-flash-exp:free

# Limit number of prompts (for quick testing)
python measure_performance.py google/gemini-2.0-flash-exp:free --max-prompts 10

# Custom output file
python measure_performance.py MODEL_NAME --output my_results.csv
```

### Multi-Model Testing
```bash
# Edit models_config.json to specify models
python run_tests.py
```

### Analysis
```bash
# Analyze single model results
python analyze_results.py results/performance_results.csv

# Compare multiple models
python analyze_results.py results/*.csv
```

## Output Files

### Results Directory (`results/`)
- `performance_*.csv`: Raw performance data with metrics per prompt
- Columns: model_name, prompt_type, input_tokens, output_tokens, time_to_first_token, throughput_tokens_per_sec, total_time, etc.

### Analysis Directory (`analysis/`)
- `performance_report.md`: Comprehensive analysis report
- `summary_statistics.json`: Numerical summaries and correlations
- Visualization PNGs:
  - `time_to_first_token_vs_input.png`: Response latency scaling
  - `throughput_vs_output.png`: Generation speed patterns
  - `throughput_distribution.png`: Speed consistency comparison
  - `performance_by_task_type.png`: Task-specific performance heatmap

## Key Metrics Measured

- **Time to First Token (TTFT)**: Responsiveness for interactive use
- **Throughput**: Tokens generated per second
- **Total Time**: End-to-end request duration
- **Token Efficiency**: Time per input/output token
- **Consistency**: Standard deviations and distributions
- **Correlations**: How performance scales with input/output size

## Configuration

Edit `models_config.json` to:
- Add/remove models to test
- Set maximum prompts per model
- Adjust request delays and parameters

## Sample Models Supported

- Google Gemini (2.0 Flash, Pro)
- Anthropic Claude (3.5 Sonnet, Haiku)
- OpenAI GPT (4o, 4o-mini)
- Meta Llama (3.2 variants)
- Qwen 2.5 series
- Any OpenRouter-supported model

## Performance Notes

- Tests are sorted by complexity (simple first) for easier debugging
- Automatic retry logic for API failures
- Respectful rate limiting between requests
- Support for vision models with base64 image encoding
- Handles edge cases like empty responses and timeouts