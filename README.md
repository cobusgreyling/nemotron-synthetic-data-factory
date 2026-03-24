# Nemotron Synthetic Data Factory

**Generate, score, filter, and export training data using NVIDIA Nemotron models.**

Over 98% of the data used to align NVIDIA's Nemotron models is synthetically generated. This repo implements that pipeline.

---

## How It Works

```
Topic + Domain ──→ Nemotron Instruct ──→ Q&A Pairs ──→ Nemotron Reward ──→ Scored Pairs ──→ Filter ──→ JSONL
                   (generate)                          (score 5 dims)      (threshold)      (export)
```

1. **Generate** — Nemotron produces diverse Q&A pairs for your topic and domain
2. **Score** — The Reward model evaluates each pair on 5 dimensions
3. **Filter** — Pairs below your quality threshold are discarded
4. **Export** — Passing pairs are saved as JSONL for fine-tuning

## The Five Scoring Dimensions

| Dimension | What It Measures |
|-----------|-----------------|
| **Helpfulness** | Does the answer address what the user actually needs? |
| **Correctness** | Is the information factually accurate? |
| **Coherence** | Does the answer flow logically without contradictions? |
| **Complexity** | Is the depth appropriate for the question? |
| **Verbosity** | Is the length appropriate — not too terse, not too padded? |

## Quick Start

```bash
# Clone and install
git clone https://github.com/cobusgreyling/nemotron-synthetic-data-factory.git
cd nemotron-synthetic-data-factory
pip install -r requirements.txt

# Set your NVIDIA API key
cp .env.example .env
# Edit .env with your key from build.nvidia.com

# Run the Gradio demo
python app.py
# Opens on http://localhost:7870

# Or run the CLI pipeline
python pipeline.py --topic "machine learning" --count 10 --threshold 2.0
```

## CLI Usage

```bash
# Basic generation
python pipeline.py --topic "customer support" --count 10

# With domain template and higher quality threshold
python pipeline.py --topic "Python debugging" --domain technical --count 15 --threshold 2.5

# With seed examples for style control
python pipeline.py --topic "investing basics" --domain finance --count 10 --seed examples/seed_ml.json

# Custom output directory
python pipeline.py --topic "data privacy" --domain legal --count 20 --output my_dataset/
```

## Domain Templates

| Domain | Focus |
|--------|-------|
| `general` | No specific guidance |
| `customer_support` | Complaints, returns, billing, troubleshooting |
| `medical` | Symptoms, treatments, prevention, when to seek care |
| `legal` | Contracts, rights, procedures, compliance |
| `technical` | Debugging, architecture, best practices, code review |
| `finance` | Budgeting, investing, taxes, accounting |
| `education` | Study techniques, curriculum design, assessment |

## Gradio Demo

The interactive demo has three tabs:

**Pipeline** — Enter a topic, pick a domain, set pair count and quality threshold. Hit generate. Watch pairs appear with live scoring. Download the filtered JSONL.

**Score a Pair** — Paste any question and answer. See how the Nemotron Reward model scores it across all 5 dimensions.

**About** — How it works, model details, and domain template descriptions.

## Output Format

The exported JSONL follows the standard fine-tuning format:

```json
{"messages": [{"role": "user", "content": "What is transfer learning?"}, {"role": "assistant", "content": "Transfer learning is..."}], "metadata": {"scores": {"helpfulness": 3.2, "correctness": 3.5, "coherence": 3.8, "complexity": 2.1, "verbosity": 2.0}, "avg_score": 2.92}}
```

Each line is a complete conversation. Metadata includes all 5 dimension scores and the average.

## Seed Examples

Provide seed examples to control the style of generated data:

```json
[
  {"question": "What is X?", "answer": "X is..."},
  {"question": "How does Y work?", "answer": "Y works by..."}
]
```

Three examples are usually enough to establish a consistent style.

## Docker

```bash
docker build -t synth-factory .
docker run -e NVIDIA_API_KEY=nvapi-xxx -p 7870:7870 synth-factory
```

## Models Used

| Role | Model | Endpoint |
|------|-------|----------|
| Generator | `nvidia/llama-3.1-nemotron-70b-instruct` | NIM API |
| Reward | `nvidia/nemotron-4-340b-reward` | NIM API |

Both models are accessed via the NVIDIA NIM API at `integrate.api.nvidia.com`. Get your API key at [build.nvidia.com](https://build.nvidia.com).

## Project Structure

```
├── app.py                 # Gradio interactive demo
├── pipeline.py            # Core pipeline + CLI
├── blog.md                # Full write-up
├── requirements.txt       # Dependencies
├── Dockerfile             # Container support
├── .env.example           # API key template
└── examples/
    └── seed_ml.json       # Sample seed examples
```

## The Bigger Picture

The synthetic data flywheel: generate → score → filter → fine-tune → repeat. Each cycle produces a better model that generates better data. NVIDIA proved this works — 98% of their alignment data is synthetic.

See [blog.md](blog.md) for the full write-up.

---

*Chief Evangelist @ Kore.ai | I'm passionate about exploring the intersection of AI and language. From Language Models, AI Agents to Agentic Applications, Development Frameworks & Data-Centric Productivity Tools, I share insights and ideas on how these technologies are shaping the future.*
