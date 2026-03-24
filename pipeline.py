"""
Synthetic Data Factory — Generate and score training data using Nemotron models.

Pipeline:
  1. Nemotron generates synthetic Q&A pairs for a given topic/domain
  2. Nemotron Reward model scores each pair on 5 dimensions
  3. Filter by quality thresholds
  4. Export as JSONL for fine-tuning

Usage:
    python pipeline.py --topic "machine learning" --count 10
    python pipeline.py --topic "customer support" --count 20 --threshold 2.0
"""

import os
import sys
import json
import time
import re
import argparse
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

GENERATOR_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
REWARD_MODEL = "nvidia/nemotron-4-340b-reward"

CLIENT = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
)

# ---------------------------------------------------------------------------
# Generation prompts
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = """You are an expert training data generator. Your task is to create high-quality question-answer pairs for fine-tuning language models.

Rules:
- Generate diverse, realistic questions that a user would actually ask
- Answers should be detailed, accurate, and well-structured
- Vary difficulty from simple to complex
- Vary question types: factual, analytical, how-to, comparison, creative
- Each pair must be self-contained — no references to other pairs
- Output ONLY valid JSON array, no extra text"""

GENERATION_PROMPT = """Generate {count} diverse question-answer pairs about: {topic}

{seed_context}

Output format — a JSON array of objects:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

Generate exactly {count} pairs. Output ONLY the JSON array."""

DOMAIN_TEMPLATES = {
    "general": "",
    "customer_support": "Focus on realistic customer service scenarios: complaints, product questions, returns, billing issues, troubleshooting.",
    "medical": "Focus on health-related questions a patient might ask. Include symptoms, treatments, prevention, and when to seek care. Always note that professional medical advice should be sought.",
    "legal": "Focus on common legal questions: contracts, rights, procedures, compliance. Always note that professional legal advice should be sought.",
    "technical": "Focus on software engineering: debugging, architecture, best practices, code review, DevOps, databases.",
    "finance": "Focus on personal and business finance: budgeting, investing, taxes, accounting principles, financial planning.",
    "education": "Focus on learning and teaching: study techniques, curriculum design, assessment methods, student engagement.",
}


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def generate_pairs(topic: str, count: int = 5, domain: str = "general",
                   seed_examples: list[dict] | None = None) -> list[dict]:
    """Generate synthetic Q&A pairs using Nemotron."""
    seed_context = ""
    if seed_examples:
        seed_context = "Use these examples as style reference (but generate NEW questions):\n"
        for ex in seed_examples[:3]:
            seed_context += f'Q: {ex["question"]}\nA: {ex["answer"]}\n\n'

    domain_hint = DOMAIN_TEMPLATES.get(domain, "")
    if domain_hint:
        seed_context = domain_hint + "\n\n" + seed_context

    prompt = GENERATION_PROMPT.format(
        count=count, topic=topic, seed_context=seed_context.strip()
    )

    try:
        response = CLIENT.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.8,
            top_p=0.95,
        )
        text = response.choices[0].message.content.strip()

        # Extract JSON array from response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            pairs = json.loads(match.group())
            return [p for p in pairs if "question" in p and "answer" in p]
        return []
    except Exception as e:
        print(f"Generation error: {e}")
        return []


def score_pair(question: str, answer: str) -> dict:
    """Score a Q&A pair using the Nemotron Reward model.

    Returns scores for: helpfulness, correctness, coherence, complexity, verbosity.
    """
    try:
        response = CLIENT.chat.completions.create(
            model=REWARD_MODEL,
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        )
        content = response.choices[0].message.content.strip()

        # Parse "helpfulness:1.6,correctness:1.6,coherence:3.3,complexity:0.5,verbosity:0.5"
        scores = {}
        for part in content.split(","):
            if ":" in part:
                key, val = part.strip().split(":", 1)
                try:
                    scores[key.strip()] = float(val.strip())
                except ValueError:
                    pass

        return scores if scores else {"raw": content}
    except Exception as e:
        return {"error": str(e)}


def score_pairs(pairs: list[dict], progress_callback=None) -> list[dict]:
    """Score all pairs and attach scores."""
    scored = []
    for i, pair in enumerate(pairs):
        if progress_callback:
            progress_callback(i, len(pairs), pair["question"][:60])
        scores = score_pair(pair["question"], pair["answer"])
        pair["scores"] = scores
        pair["avg_score"] = _avg_score(scores)
        scored.append(pair)
    return scored


def filter_pairs(pairs: list[dict], threshold: float = 1.5) -> list[dict]:
    """Filter pairs by average score threshold."""
    return [p for p in pairs if p.get("avg_score", 0) >= threshold]


def export_jsonl(pairs: list[dict], output_path: str) -> str:
    """Export pairs as JSONL for fine-tuning."""
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            entry = {
                "messages": [
                    {"role": "user", "content": pair["question"]},
                    {"role": "assistant", "content": pair["answer"]},
                ]
            }
            if "scores" in pair:
                entry["metadata"] = {"scores": pair["scores"], "avg_score": pair.get("avg_score", 0)}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return output_path


def export_json_report(pairs: list[dict], metadata: dict, output_path: str) -> str:
    """Export full report with metadata and scores."""
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": metadata,
        "summary": _summarise(pairs),
        "pairs": pairs,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCORE_KEYS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


def _avg_score(scores: dict) -> float:
    vals = [scores[k] for k in SCORE_KEYS if k in scores]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _summarise(pairs: list[dict]) -> dict:
    if not pairs:
        return {}
    totals = {k: 0.0 for k in SCORE_KEYS}
    counted = 0
    for p in pairs:
        s = p.get("scores", {})
        if any(k in s for k in SCORE_KEYS):
            for k in SCORE_KEYS:
                totals[k] += s.get(k, 0)
            counted += 1
    if counted == 0:
        return {"total_pairs": len(pairs)}
    return {
        "total_pairs": len(pairs),
        "scored_pairs": counted,
        "avg_scores": {k: round(v / counted, 3) for k, v in totals.items()},
        "overall_avg": round(sum(totals.values()) / (counted * len(SCORE_KEYS)), 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_pipeline(topic: str, count: int = 5, domain: str = "general",
                 threshold: float = 1.5, output_dir: str = "output",
                 seed_examples: list[dict] | None = None) -> dict:
    """Run the full synthetic data pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^a-zA-Z0-9]+', '_', topic)[:40]

    print(f"\n{'='*60}")
    print(f"  Nemotron Synthetic Data Factory")
    print(f"  Topic: {topic}")
    print(f"  Domain: {domain}")
    print(f"  Requested pairs: {count}")
    print(f"  Quality threshold: {threshold}")
    print(f"{'='*60}\n")

    # Step 1: Generate
    print("[1/4] Generating synthetic Q&A pairs...")
    t0 = time.time()
    pairs = generate_pairs(topic, count, domain, seed_examples)
    gen_time = time.time() - t0
    print(f"  Generated {len(pairs)} pairs in {gen_time:.1f}s\n")

    if not pairs:
        print("  No pairs generated. Check your API key and try again.")
        return {"error": "No pairs generated"}

    # Step 2: Score
    print("[2/4] Scoring with Nemotron Reward model...")
    t0 = time.time()

    def progress(i, total, preview):
        print(f"  [{i+1}/{total}] Scoring: {preview}...")

    scored = score_pairs(pairs, progress_callback=progress)
    score_time = time.time() - t0
    print(f"  Scored {len(scored)} pairs in {score_time:.1f}s\n")

    # Step 3: Filter
    print(f"[3/4] Filtering (threshold >= {threshold})...")
    filtered = filter_pairs(scored, threshold)
    print(f"  {len(filtered)}/{len(scored)} pairs passed quality filter\n")

    # Step 4: Export
    print("[4/4] Exporting...")
    jsonl_path = export_jsonl(
        filtered, os.path.join(output_dir, f"synth_{safe_topic}_{timestamp}.jsonl")
    )
    report_path = export_json_report(
        scored,
        {
            "topic": topic,
            "domain": domain,
            "generator_model": GENERATOR_MODEL,
            "reward_model": REWARD_MODEL,
            "threshold": threshold,
            "timestamp": timestamp,
            "generation_time_s": round(gen_time, 2),
            "scoring_time_s": round(score_time, 2),
        },
        os.path.join(output_dir, f"report_{safe_topic}_{timestamp}.json"),
    )

    summary = _summarise(scored)
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"  Generated: {len(pairs)} | Passed filter: {len(filtered)}")
    if summary.get("avg_scores"):
        print(f"  Avg scores: {summary['avg_scores']}")
    print(f"  JSONL:   {jsonl_path}")
    print(f"  Report:  {report_path}")
    print(f"{'='*60}\n")

    return {
        "pairs_generated": len(pairs),
        "pairs_passed": len(filtered),
        "summary": summary,
        "jsonl_path": jsonl_path,
        "report_path": report_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Nemotron Synthetic Data Factory")
    parser.add_argument("--topic", type=str, required=True, help="Topic to generate data about")
    parser.add_argument("--count", type=int, default=5, help="Number of Q&A pairs to generate")
    parser.add_argument("--domain", type=str, default="general",
                        choices=list(DOMAIN_TEMPLATES.keys()), help="Domain template")
    parser.add_argument("--threshold", type=float, default=1.5, help="Minimum avg score to keep")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--seed", type=str, help="Path to seed examples JSON file")
    args = parser.parse_args()

    seed_examples = None
    if args.seed:
        with open(args.seed) as f:
            seed_examples = json.load(f)

    run_pipeline(args.topic, args.count, args.domain, args.threshold, args.output, seed_examples)


if __name__ == "__main__":
    main()
