# Pharma AI Monitor — LLM Output Quality Scoring for Drug Information

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green?logo=openai) ![FDA](https://img.shields.io/badge/data-openFDA-red) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

Production-ready pipeline for monitoring and scoring AI-generated pharmaceutical information against verified drug label claims. Fetches real prescribing information from **openFDA**, generates LLM answers, and evaluates them across multiple judge strategies — catching hallucinations, factual drift, and unsupported claims before they reach end users.

**Demo use case:** Wegovy (semaglutide) PI claim verification — baseline vs improved scorer comparison.

---

## Architecture

```
openFDA API
(Wegovy / drug PI claims)
         │
         ▼
┌────────────────────────┐
│   generate_answers.py  │  LLM generates answers to
│   (GPT-4o / any model) │  drug-related questions
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────────────────────────┐
│              Judge Layer                   │
│                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ LLM Judge│  │ NLI Judge│  │ Entity   │ │
│  │(GPT-4o)  │  │(sentence │  │ Judge    │ │
│  │          │  │ BERT)    │  │          │ │
│  └──────────┘  └──────────┘  └──────────┘ │
│  ┌──────────┐  ┌──────────┐               │
│  │ Zero-shot│  │ Client   │               │
│  │ Classifier│ │ Method   │               │
│  └──────────┘  └──────────┘               │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│   synthesize.py          │  Aggregates judge scores
│   compare_methods.py     │  Baseline vs improved comparison
│   report.py              │  HTML dashboard output
└──────────────────────────┘
               │
               ▼
       pharma_scoring_report.html
```

---

## Key Features

- **Multi-judge ensemble** — 5 independent scoring strategies (LLM-as-judge, NLI sentence similarity, entity extraction, zero-shot classification, client-defined rubric)
- **openFDA integration** — live PI fetching for any approved drug; no manual copy-paste of label text
- **Baseline vs improved comparison** — side-by-side scoring to quantify prompt or model improvements
- **Hallucination detection** — flags unsupported claims, contraindication omissions, and dosage errors
- **HTML report dashboard** — self-contained output with per-row scores, judge breakdowns, and summary statistics
- **Dry-run mode** — `--dry-run` skips API calls for CI/testing
- **Modular judge system** — swap or add judges without changing the pipeline

---

## Project Structure

```
├── pharma_monitor/
│   ├── main.py                # CLI entry point
│   ├── config.py              # Models, thresholds, drug targets
│   ├── generate_answers.py    # LLM answer generation from PI claims
│   ├── judge.py               # Judge orchestrator
│   ├── judge_llm.py           # GPT-4o LLM-as-judge
│   ├── judge_sentence_nli.py  # Sentence-BERT NLI similarity
│   ├── judge_entity.py        # Named entity extraction judge
│   ├── judge_zeroshot.py      # Zero-shot classification judge
│   ├── judge_client_method.py # Custom client rubric judge
│   ├── synthesize.py          # Score aggregation
│   ├── compare_methods.py     # Baseline vs improved comparison
│   └── report.py              # HTML report generation
├── pharma_demo/
│   ├── run_demo.py            # Wegovy PI demo (before/after scoring)
│   ├── baseline_scorer.py     # Baseline scoring logic
│   ├── improved_scorer.py     # Improved scoring logic
│   └── demo_data.py           # Sample drug Q&A rows
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/pharma-ai-monitor.git
cd pharma-ai-monitor
pip install -r requirements.txt

# Set your OpenAI key
export OPENAI_API_KEY=your_key_here

# Run Wegovy demo (fetches live FDA data, runs both scorers, outputs HTML)
python pharma_demo/run_demo.py

# Run full monitor pipeline
python pharma_monitor/main.py

# Dry run (no API calls)
python pharma_monitor/main.py --dry-run

# Rebuild report only (skip generation + synthesis)
python pharma_monitor/main.py --report
```

---

## Judge Strategies

| Judge | Method | Best for |
|-------|--------|----------|
| `judge_llm.py` | GPT-4o scores factual accuracy 0-1 | General claim verification |
| `judge_sentence_nli.py` | Sentence-BERT NLI entailment | Paraphrase / semantic drift |
| `judge_entity.py` | Entity extraction match | Dosage, drug name, indication accuracy |
| `judge_zeroshot.py` | Zero-shot classification | Tone, safety, regulatory framing |
| `judge_client_method.py` | Custom rubric | Client-defined quality criteria |

---

## Sample Output

```
Drug: Wegovy (semaglutide 2.4 mg)
Question: What is the approved indication?

Baseline answer score:  0.61  [LLM: 0.55 | NLI: 0.68 | Entity: 0.62]
Improved answer score:  0.89  [LLM: 0.91 | NLI: 0.87 | Entity: 0.88]
Delta: +0.28  ✅ Improvement confirmed

Flags: baseline omitted "chronic weight management" qualifier
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM generation + judging | OpenAI GPT-4o |
| NLI similarity | sentence-transformers |
| FDA data | openFDA REST API |
| Report generation | Jinja2 / inline HTML |
| CLI | argparse |

---

## Use Cases

- **Pharma / medtech companies** — monitor AI assistants for drug information accuracy before deployment
- **Clinical AI auditing** — automated QA layer for LLM outputs in regulated environments
- **Prompt engineering teams** — measure factual improvement across model or prompt versions
- **Regulatory compliance** — evidence of output monitoring for EU AI Act Article 9 risk management

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany. Background in clinical data science, pharmacovigilance ML, and AI system evaluation.

---

## License

MIT
