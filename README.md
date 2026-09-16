# Fine-Tuning Qwen2.5-1.5B-Instruct Using AraFinNews Dataset

## Problem Definition

Arabic financial news is published as unstructured, free-form text, which makes it costly to extract structured financial intelligence (companies, people, countries, locations, financial events, financial metrics, sentiment) along with an accurate English translation. Typically this job requires either large general purpose LLMs (expensive and slow to serve) or rule-based NLP pipelines (fragile and hard to maintain)

This project fine-tunes a small efficient model (**Qwen2.5-1.5B-Instruct**) with **LoRA** to reliably output a fixed-schema JSON output for Arabic financial articles making structured financial intelligence extraction cheap enough to run in production while matching the extraction quality of much larger models

## Dataset Filtering, Synthetic Supervision, Validation

<img src="https://drive.google.com/uc?id=1WAxAD7baWPBlPiyVu_1s7BBUiypdagPL" alt="">

**_NOTE:_**  Although using two teachers one for extraction and the other one for translation may results in better labeling, I will use one teacher here as the first trial

## Fine-Tuning Process

<img src="https://drive.google.com/uc?id=1W3QP2z1W5Y1Og_aQ3wENCSXZCB4OFA50" alt="">

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Method:** LoRA (rank 16, all linear layers) via **LLaMA-Factory** SFT
- **Data:** 2,500 synthetically labeled Arabic financial news articles (AraFinNews)
- **Sequence length:** 3,500 tokens (cutoff)
- **Experiment tracking:** **Weights & Biases (W&B)** for loss curves and run comparison

## Evaluation

A custom `FinancialIntelligenceEvaluator` scores model outputs against the gold test set on:

- **Entity F1** — companies, people, countries, locations
- **Event F1** — `(event_type, company, percentage)` extraction
- **Metric F1** — `(metric, value, currency)` extraction
- **Translation chrF** — English translation quality

| Metric | Base | Fine-Tuned | Δ | Rel. % |
|---|---:|---:|---:|---:|
| Entity F1 | 0.5054 | 0.9205 | +0.4151 | +82.1% |
| Event F1 | 0.0667 | 0.6325 | +0.5658 | +848.3% |
| Metric F1 | 0.1576 | 0.8387 | +0.6811 | +432.2% |
| Translation chrF | 47.43 | 93.73 | +46.30 | +97.6% |

```bash
python financial_intelligence_evaluator.py \
  --gold data/test.jsonl --pred data/ft.jsonl --base-pred data/base.jsonl
```

## Benchmarking

A custom `FinancialIntelligenceBenchmarking` class measures **TTFT**, **latency**, and **throughput** (mean, median, P50/P90/P95/P99)

1. **HuggingFace `transformers`** -> base model + LoRA adapter, single-request baseline
2. **KServe + vLLM** -> the same adapter served behind an OpenAI-compatible endpoint with continuous batching

| Metric | HF + LoRA | KServe + vLLM | Speedup |
|---|---:|---:|---:|
| TTFT (p50) | 1.85 s | 0.17 s | 10.9× |
| TTFT (p95) | 2.40 s | 0.29 s | 8.3× |
| Total latency (p50) | 14.2 s | 3.1 s | 4.6× |
| Output tokens/sec (median) | 26 tok/s | 132 tok/s | 5.1× |
| System output tokens/sec (concurrency 8) | 26 tok/s | 890 tok/s | 34× |
| Requests/sec (concurrency 8) | 0.07 | 1.9 | 27× |

```bash
python financial_intelligence_benchmarking.py \
  --data data/test.jsonl --kserve-url http://qwen-arafinnews.default.example.com --concurrency 8
```

## Deployment: KServe + vLLM

- The fine-tuned LoRA adapter is served via the **vLLM** serving runtime with `--enable-lora`, enabling **continuous batching** and **PagedAttention** for high-throughput generation.
- Deployed as a **KServe `LLMInferenceService`** exposing an OpenAI-compatible `/openai/v1/chat/completions` route with autoscaling managed by Kubernetes


## Tech Stack

`Transformers` · `PEFT (LoRA)` · `LLaMA-Factory` · `Weights & Biases` · `vLLM` · `KServe`