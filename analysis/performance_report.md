# OpenRouter Model Performance Analysis Report
============================================================

## Executive Summary

**Model Tested:** google/gemini-2.0-flash-exp:free
**Total Tests:** 4
**Average Throughput:** 1208.61 tokens/sec
**Average Time to First Token:** 1.929 seconds
**Median Throughput:** 1242.12 tokens/sec

## Detailed Performance Metrics

### google/gemini-2.0-flash-exp:free

**Basic Metrics:**
- Total tests completed: 4
- Average input tokens: 11
- Average output tokens: 24

**Speed Metrics:**
- Average time to first token: 1.929s (±0.325s)
- Average time per input token: 0.2041s
- Average throughput: 1208.61 tokens/sec (±884.01)
- Median throughput: 1242.12 tokens/sec
- Throughput range: 100.38 - 2249.83 tokens/sec
- Average total time: 2.02s
- Average time per total token: 0.0680s

**Correlations:**
- Time to first token correlation with input tokens: 0.479
- Throughput correlation with output tokens: 0.201
- Total time correlation with total tokens: 0.893

## Key Findings

**Performance Characteristics:**
- High throughput model - excellent for bulk text generation
- Moderate response time - suitable for most applications
- Time to first token shows moderate scaling with input length

## Recommendations

- This model shows excellent performance characteristics for most applications
- High throughput variability - performance may depend significantly on task type