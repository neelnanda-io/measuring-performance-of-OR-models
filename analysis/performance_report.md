# OpenRouter Model Performance Analysis Report
============================================================

## Executive Summary

**Models Compared:**
- google/gemini-2.5-flash-lite-preview-06-17: 1113.88 tokens/sec avg throughput
- google/gemini-2.5-flash-preview-05-20: 2395.61 tokens/sec avg throughput
- google/gemini-2.5-pro: 105.90 tokens/sec avg throughput

## Detailed Performance Metrics

### google/gemini-2.5-flash-lite-preview-06-17

**Basic Metrics:**
- Total tests completed: 78
- Average input tokens: 37573
- Average output tokens: 3490

**Speed Metrics:**
- Average time to first token: 0.005s (±0.013s)
- Average time per input token: 0.0002s
- Average throughput: 1113.88 tokens/sec (±6550.38)
- Median throughput: 317.44 tokens/sec
- Throughput range: 0.01 - 58126.75 tokens/sec
- Average total time: 10.17s
- Average time per total token: 0.1455s

**Correlations:**
- Time to first token correlation with input tokens: 0.697
- Throughput correlation with output tokens: -0.026
- Total time correlation with total tokens: 0.209

### google/gemini-2.5-flash-preview-05-20

**Basic Metrics:**
- Total tests completed: 75
- Average input tokens: 32142
- Average output tokens: 1135

**Speed Metrics:**
- Average time to first token: 0.266s (±2.264s)
- Average time per input token: 0.0002s
- Average throughput: 2395.61 tokens/sec (±13095.33)
- Median throughput: 145.11 tokens/sec
- Throughput range: 1.90 - 108473.38 tokens/sec
- Average total time: 12.73s
- Average time per total token: 0.0047s

**Correlations:**
- Time to first token correlation with input tokens: 0.741
- Throughput correlation with output tokens: -0.062
- Total time correlation with total tokens: 0.088

### google/gemini-2.5-pro

**Basic Metrics:**
- Total tests completed: 73
- Average input tokens: 25008
- Average output tokens: 651

**Speed Metrics:**
- Average time to first token: 7.879s (±7.198s)
- Average time per input token: 0.6061s
- Average throughput: 105.90 tokens/sec (±219.70)
- Median throughput: 31.34 tokens/sec
- Throughput range: 0.95 - 1132.00 tokens/sec
- Average total time: 15.17s
- Average time per total token: 0.0567s

**Correlations:**
- Time to first token correlation with input tokens: -0.055
- Throughput correlation with output tokens: -0.024
- Total time correlation with total tokens: -0.018

## Key Findings

**Model Comparison:**
- **Highest throughput:** google/gemini-2.5-flash-preview-05-20 (2395.61 tokens/sec)
- **Fastest response:** google/gemini-2.5-flash-lite-preview-06-17 (0.005s)

## Recommendations

- Choose models based on your specific use case:
  - google/gemini-2.5-flash-lite-preview-06-17: Best for high-volume text generation
  - google/gemini-2.5-flash-preview-05-20: Best for high-volume text generation
  - google/gemini-2.5-pro: Best for high-volume text generation