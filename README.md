# OpenRouter Model Performance Measurement

A comprehensive benchmarking tool for measuring the performance characteristics of OpenRouter language models across various task types and input/output sizes.

## Features

- **🚀 Async Parallel Testing**: Up to 5x faster with concurrent requests and smart retry logic
- **📊 Comprehensive Testing**: 70+ diverse prompts across 17+ categories
- **🖼️ Multi-Modal Support**: Text, code, math, vision tasks, and long context
- **⚡ Detailed Metrics**: Time to first token, throughput, total response time
- **🔄 Smart Error Handling**: Automatic retry for rate limits, robust filtering
- **🎯 Batch Testing**: Test multiple models automatically
- **📈 Rich Analysis**: Statistical summaries, correlations, and visualizations
- **📄 Professional Reports**: Automated markdown reports with key insights

## Test Categories

- **Simple Tasks**: Basic Q&A, definitions
- **Mathematical Problems**: From arithmetic to complex proofs
- **Coding Challenges**: Simple functions to complex algorithms
- **Creative Writing**: Haikus to full novellas
- **Document Processing**: Summarization of 1K-100K token documents
- **Long Context**: Repetitive text recognition up to 900K tokens
- **High Output**: Number sequences, encyclopedia entries
- **Vision Tasks**: Geometric shape recognition (1-5 images per prompt)
- **🚨 Extreme Scale Tasks** (requires `--include-long`):
  - **Repetitive Word**: 250K, 500K, 900K token inputs
  - **Repeat After Me**: 10K-450K token sequences using simple dictionary words

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt` 
3. Set your OpenRouter API key (choose one method):
   ```bash
   # Method 1: Environment variable
   export OPENROUTER_API_KEY=your_key_here
   
   # Method 2: Create api.secrets file
   echo "OPENROUTER_API_KEY=your_key_here" > api.secrets
   ```
4. Generate test data:
   ```bash
   python generate_prompts.py
   python generate_images.py
   ```

## Usage

### 🚀 Fast Async Testing (Recommended)

**Single Model:**
```bash
# Fast parallel testing (5x faster)
python measure_performance_async.py "google/gemini-2.0-flash-exp:free"

# Conservative concurrency (safer for rate limits)
python measure_performance_async.py "google/gemini-2.0-flash-exp:free" --max-concurrent 3

# Quick test with limited prompts
python measure_performance_async.py "google/gemini-2.0-flash-exp:free" --max-prompts 10
```

**Multi-Model Testing:**
```bash
# Test multiple models automatically
python run_async_tests.py
```

### 🐌 Original Serial Testing
```bash
# Slower but ultra-safe (20s delay between requests)
python measure_performance.py "google/gemini-2.0-flash-exp:free"
```

### 📊 Key Performance Options

| Flag | Purpose | Recommended Values |
|------|---------|-------------------|
| `--max-concurrent` | Parallel requests | 3-5 (conservative), 10+ (aggressive) |
| `--max-prompts` | Limit test size | 10 (quick), 50+ (comprehensive) |
| `--include-long` | Include >50K token tasks | ⚠️ **WARNING: Very expensive!** |
| `--output` | Custom filename | Auto-generated with model name + timestamp |

### ⚠️ **Long Tasks Warning**
The `--include-long` flag includes extreme-scale tasks with 50K-900K tokens. These can be **very expensive** (potentially hundreds of dollars per test) and may hit model context limits. Use with caution!

### Analysis

**Command Line Analysis:**
```bash
# Analyze single model results
python analyze_results.py results/performance_results.csv

# Compare multiple models
python analyze_results.py results/*.csv
```

**Interactive Analysis (VS Code):**
```bash
# Quick exploration with inline graphs
code interactive_analysis.py  # Run cells with Ctrl+Enter

# Lightweight exploration
code quick_explore.py

# Advanced statistical analysis
code advanced_analysis.py
```

## Output Files

### Results Directory (`results/`)
- `performance_*.csv`: Raw performance data with metrics per prompt
- Columns: model_name, prompt_type, input_tokens, output_tokens, time_to_first_token, throughput_tokens_per_sec, total_time, etc.

### Analysis Directory (`analysis/`)

**Automated Reports:**
- `performance_report.md`: Comprehensive analysis report
- `summary_statistics.json`: Numerical summaries and correlations
- Visualization PNGs:
  - `time_to_first_token_vs_input.png`: Response latency scaling
  - `throughput_vs_output.png`: Generation speed patterns
  - `throughput_distribution.png`: Speed consistency comparison
  - `performance_by_task_type.png`: Task-specific performance heatmap

**Interactive Analysis Outputs:**
- `processed_results.csv`: Enhanced dataset with calculated metrics
- `interactive_statistics.json`: Detailed statistics from interactive analysis
- `comprehensive_analysis.csv`: Full dataset with advanced features
- `model_comparison.json`: Side-by-side model performance comparison

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

## Interactive Analysis Features

### VS Code Jupyter-Style Analysis
The project includes three interactive Python files optimized for VS Code's Python Interactive mode:

1. **`interactive_analysis.py`** - Comprehensive step-by-step analysis
   - Load and process multiple result files
   - Generate publication-quality visualizations
   - Calculate detailed statistics and correlations
   - Customizable analysis sections

2. **`quick_explore.py`** - Lightweight rapid exploration
   - Quick model performance overview
   - Instant throughput and latency plots
   - Task-specific performance breakdown
   - Efficiency calculations

3. **`advanced_analysis.py`** - Statistical deep-dive
   - Statistical significance testing between models
   - Performance clustering and PCA analysis
   - Task complexity scoring
   - Performance scaling analysis
   - Correlation heatmaps

### Usage in VS Code
1. Open any `.py` file above in VS Code
2. Use `Ctrl+Enter` to run individual cells
3. Graphs appear in the Interactive window
4. Modify variables and re-run for custom analysis
5. Export results for further processing

## 🔧 Smart Error Handling & Retry Logic

### Automatic Retries
- **Rate limit errors (429)**: Wait 20 seconds, retry up to 4 times
- **Temporary failures**: Same retry logic for network issues
- **Non-recoverable errors**: Fail immediately (auth, model not found, etc.)

### Robust Data Filtering
The analysis automatically excludes:
- ❌ Failed requests (`success != True`)
- ❌ Zero output token responses
- ❌ Invalid timing data (`total_time <= 0`)
- ❌ Corrupted throughput calculations

### Output File Naming
Default filenames include model name and timestamp for easy identification:
```bash
# Examples:
google_gemini-2.0-flash-exp_free_20250123_143027.csv
anthropic_claude-3-5-sonnet_20250123_143152.csv
```

## 📊 Performance Analysis Results

### Key Performance Benchmarks (Based on Recent Testing)

| Model | Avg Throughput | TTFT (Short) | TTFT (Long) | Best Use Case |
|-------|---------------|--------------|-------------|---------------|
| **Gemini 2.5 Flash** | **2,396 tok/s** | 0.266s | Variable | High-volume generation |
| **Gemini 2.5 Flash Lite** | 1,114 tok/s | **0.005s** | Low latency | Real-time applications |
| **Gemini 2.5 Pro** | 106 tok/s | 7.879s | Consistent | Quality-focused tasks |

### Scaling Characteristics

**Time to First Token (TTFT) vs Input Size:**
- **Flash Lite**: Strong positive correlation (0.697) - TTFT increases predictably with input length
- **Flash**: Strong positive correlation (0.741) - Similar scaling pattern
- **Pro**: Weak correlation (-0.055) - More consistent regardless of input size

**Throughput vs Output Size:**
- All models show weak negative correlation with output size (-0.024 to -0.062)
- Throughput remains relatively stable across different output lengths
- Flash models show higher variance but better peak performance

**Linear Regression Coefficients:**
```
TTFT = β₀ + β₁ × input_tokens
- Flash Lite: β₁ ≈ 0.0002 (2ms per 10K input tokens)
- Flash: β₁ ≈ 0.0002 (similar scaling)
- Pro: β₁ ≈ 0.6061 (much higher baseline latency)
```

### Model Selection Guidelines

🚀 **For Real-time Applications** (chatbots, interactive tools):
- **Choose**: Gemini 2.5 Flash Lite
- **Why**: Sub-10ms TTFT, good throughput, excellent responsiveness

⚡ **For High-volume Batch Processing**:
- **Choose**: Gemini 2.5 Flash
- **Why**: Highest throughput (2.4K tok/s), efficient for large workloads

🎯 **For Quality-critical Tasks** (analysis, reasoning):
- **Choose**: Gemini 2.5 Pro
- **Why**: Consistent performance, better reasoning capabilities

## 🔧 Adapting to Google Developer Gemini API

### Required Code Changes

To use Google's official Gemini API instead of OpenRouter:

1. **Install Google AI Python SDK:**
   ```bash
   pip install google-generativeai
   ```

2. **Update API Configuration** in `measure_performance_async.py` and `measure_performance.py`:
   ```python
   # Replace OpenRouter imports:
   # import openai
   
   # With Google AI imports:
   import google.generativeai as genai
   
   # Replace OpenRouter client initialization:
   # client = openai.AsyncOpenAI(...)
   
   # With Google AI configuration:
   genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
   model = genai.GenerativeModel('gemini-2.5-flash')
   ```

3. **Update Model Names** in your test configurations:
   ```python
   # OpenRouter format:
   "google/gemini-2.5-flash-preview-05-20"
   
   # Google AI format:
   "gemini-2.5-flash"
   "gemini-2.5-pro"
   ```

4. **Modify API Calls** (around line 100 in async script):
   ```python
   # Replace OpenRouter chat completion:
   # response = await client.chat.completions.create(...)
   
   # With Google AI generate_content:
   response = await model.generate_content_async(prompt_text)
   ```

5. **Update API Key Environment Variable:**
   ```bash
   # Instead of:
   export OPENROUTER_API_KEY=your_key
   
   # Use:
   export GOOGLE_API_KEY=your_google_api_key
   ```

### Getting Google API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key" → "Create API Key"
3. Copy the generated key to your environment

### Rate Limits & Pricing

- **Google Direct API**: Often higher rate limits than OpenRouter
- **Pricing**: Check [Google AI Pricing](https://ai.google.dev/pricing) for current rates
- **Free Tier**: Available with generous quotas for testing

## Performance Notes

- Tests are sorted by complexity (simple first) for easier debugging
- OpenRouter requests include app identification ("Performance Measurement Tool")
- Improved token estimation (more accurate than simple word counting)
- Support for vision models with base64 image encoding
- Handles edge cases like empty responses and timeouts
- Concurrent request management with semaphore-based rate limiting