"""
Synthetic Data Factory — Interactive Gradio Demo
Generate, score, filter, and export synthetic training data using Nemotron models.
"""

import os
import sys
import json
import time
import html as htmlmod

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr
from pipeline import (
    generate_pairs, score_pair, filter_pairs, export_jsonl, export_json_report,
    SCORE_KEYS, DOMAIN_TEMPLATES, GENERATOR_MODEL, REWARD_MODEL, _avg_score, _summarise,
)

PORT = 7870

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

NVIDIA_THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f0fdf4", c100="#dcfce7", c200="#bbf7d0", c300="#86efac",
        c400="#4ade80", c500="#76b900", c600="#65a30d", c700="#4d7c0f",
        c800="#3f6212", c900="#365314", c950="#1a2e05",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0", c300="#cbd5e1",
        c400="#94a3b8", c500="#64748b", c600="#475569", c700="#334155",
        c800="#1e293b", c900="#0f172a", c950="#020617",
    ),
    font=["Inter", "system-ui", "sans-serif"],
    font_mono=["JetBrains Mono", "Fira Code", "monospace"],
)

CSS = """
.banner-container {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d1117 100%);
    border: 1px solid #76b900;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 16px;
    text-align: center;
}
.banner-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #e6edf3;
    margin-bottom: 6px;
}
.nvidia-badge {
    background: #76b900;
    color: #000;
    padding: 2px 10px;
    border-radius: 4px;
    font-weight: 900;
    font-size: 0.85em;
    letter-spacing: 1px;
    margin-right: 8px;
}
.banner-subtitle {
    font-size: 0.95rem;
    color: #8b949e;
}
.score-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
}
.score-label {
    font-weight: 700;
    color: #76b900;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.score-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #e6edf3;
}
.score-bar {
    height: 6px;
    border-radius: 3px;
    margin-top: 4px;
}
.pair-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    margin: 10px 0;
}
.pair-question {
    color: #58a6ff;
    font-weight: 600;
    margin-bottom: 8px;
}
.pair-answer {
    color: #c9d1d9;
    line-height: 1.6;
}
.pair-scores {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #21262d;
}
.pass-badge {
    background: #238636;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 700;
}
.fail-badge {
    background: #da3633;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 700;
}
.stats-summary {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}
"""


def esc(text: str) -> str:
    return htmlmod.escape(str(text))


def score_color(val: float) -> str:
    if val >= 3.0:
        return "#3fb950"
    if val >= 2.0:
        return "#76b900"
    if val >= 1.0:
        return "#d29922"
    return "#f85149"


def score_bar_html(label: str, value: float, max_val: float = 4.0) -> str:
    pct = min(100, max(0, (value / max_val) * 100))
    color = score_color(value)
    return (
        f'<div style="margin:4px 0;">'
        f'<div style="display:flex;justify-content:space-between;">'
        f'<span style="color:#8b949e;font-size:0.8rem;">{esc(label)}</span>'
        f'<span style="color:{color};font-family:monospace;font-size:0.85rem;">{value:.2f}</span>'
        f'</div>'
        f'<div style="background:#21262d;border-radius:3px;height:6px;margin-top:2px;">'
        f'<div style="background:{color};width:{pct}%;height:6px;border-radius:3px;"></div>'
        f'</div></div>'
    )


def pair_card_html(pair: dict, index: int, threshold: float) -> str:
    scores = pair.get("scores", {})
    avg = pair.get("avg_score", 0)
    passed = avg >= threshold
    badge = '<span class="pass-badge">PASS</span>' if passed else '<span class="fail-badge">FAIL</span>'

    score_bars = ""
    for key in SCORE_KEYS:
        if key in scores:
            score_bars += score_bar_html(key.capitalize(), scores[key])

    return (
        f'<div class="pair-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
        f'<span style="color:#8b949e;font-size:0.8rem;">Pair #{index + 1}</span>'
        f'{badge} <span style="font-family:monospace;color:#8b949e;font-size:0.8rem;">avg: {avg:.2f}</span>'
        f'</div>'
        f'<div class="pair-question">Q: {esc(pair["question"])}</div>'
        f'<div class="pair-answer">{esc(pair["answer"][:500])}{"..." if len(pair.get("answer", "")) > 500 else ""}</div>'
        f'<div class="pair-scores">{score_bars}</div>'
        f'</div>'
    )


def summary_html(pairs: list[dict], threshold: float, gen_time: float, score_time: float) -> str:
    summary = _summarise(pairs)
    passed = len([p for p in pairs if p.get("avg_score", 0) >= threshold])
    total = len(pairs)

    avg_html = ""
    for key in SCORE_KEYS:
        val = summary.get("avg_scores", {}).get(key, 0)
        avg_html += score_bar_html(key.capitalize(), val)

    return (
        f'<div class="stats-summary">'
        f'<div style="display:flex;gap:30px;margin-bottom:16px;">'
        f'<div><span style="color:#8b949e;font-size:0.8rem;">Generated</span><br>'
        f'<span style="color:#e6edf3;font-size:1.4rem;font-weight:700;">{total}</span></div>'
        f'<div><span style="color:#8b949e;font-size:0.8rem;">Passed</span><br>'
        f'<span style="color:#3fb950;font-size:1.4rem;font-weight:700;">{passed}</span></div>'
        f'<div><span style="color:#8b949e;font-size:0.8rem;">Failed</span><br>'
        f'<span style="color:#f85149;font-size:1.4rem;font-weight:700;">{total - passed}</span></div>'
        f'<div><span style="color:#8b949e;font-size:0.8rem;">Gen Time</span><br>'
        f'<span style="color:#e6edf3;font-size:1.4rem;font-weight:700;">{gen_time:.1f}s</span></div>'
        f'<div><span style="color:#8b949e;font-size:0.8rem;">Score Time</span><br>'
        f'<span style="color:#e6edf3;font-size:1.4rem;font-weight:700;">{score_time:.1f}s</span></div>'
        f'</div>'
        f'<div style="margin-top:12px;"><b style="color:#76b900;">Average Scores</b>{avg_html}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def run_factory(topic, count, domain, threshold, seed_text):
    if not topic.strip():
        yield "", "", "", None
        return

    yield "Generating synthetic Q&A pairs...", "", "", None

    seed_examples = None
    if seed_text.strip():
        try:
            seed_examples = json.loads(seed_text)
        except json.JSONDecodeError:
            pass

    # Generate
    t0 = time.time()
    pairs = generate_pairs(topic, int(count), domain, seed_examples)
    gen_time = time.time() - t0

    if not pairs:
        yield "No pairs generated. Check your API key.", "", "", None
        return

    yield f"Generated {len(pairs)} pairs in {gen_time:.1f}s. Now scoring...", "", "", None

    # Score each pair
    t0 = time.time()
    for i, pair in enumerate(pairs):
        scores = score_pair(pair["question"], pair["answer"])
        pair["scores"] = scores
        pair["avg_score"] = _avg_score(scores)

        # Build progressive output
        cards = ""
        for j, p in enumerate(pairs[:i+1]):
            cards += pair_card_html(p, j, threshold)

        status = f"Scoring pair {i+1}/{len(pairs)}..."
        yield status, cards, "", None

    score_time = time.time() - t0

    # Final output
    cards = ""
    for j, p in enumerate(pairs):
        cards += pair_card_html(p, j, threshold)

    stats = summary_html(pairs, threshold, gen_time, score_time)

    # Export JSONL
    filtered = filter_pairs(pairs, threshold)
    if filtered:
        jsonl_path = f"/tmp/synth_export_{int(time.time())}.jsonl"
        export_jsonl(filtered, jsonl_path)
        yield f"Done! {len(filtered)}/{len(pairs)} pairs passed (threshold {threshold})", cards, stats, jsonl_path
    else:
        yield f"Done! 0/{len(pairs)} pairs passed threshold {threshold}", cards, stats, None


# ---------------------------------------------------------------------------
# Score a single Q&A pair
# ---------------------------------------------------------------------------

def score_single(question, answer):
    if not question.strip() or not answer.strip():
        return ""

    scores = score_pair(question, answer)
    avg = _avg_score(scores)

    html = '<div class="pair-card">'
    html += f'<div style="margin-bottom:8px;"><b style="color:#76b900;">Average: {avg:.2f}</b></div>'
    for key in SCORE_KEYS:
        if key in scores:
            html += score_bar_html(key.capitalize(), scores[key])
    html += '</div>'
    return html


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EXAMPLE_SEEDS = json.dumps([
    {"question": "What is transfer learning?", "answer": "Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a different task."},
    {"question": "When should I use a CNN vs RNN?", "answer": "Use CNNs for spatial data like images. Use RNNs for sequential data like text or time series. Transformers have largely replaced RNNs for text tasks."}
], indent=2)


def build_ui():
    with gr.Blocks(title="Synthetic Data Factory — Nemotron") as demo:
        gr.HTML(
            '<div class="banner-container">'
            '<div class="banner-title"><span class="nvidia-badge">NVIDIA</span> Synthetic Data Factory</div>'
            '<div class="banner-subtitle">Generate · Score · Filter · Export training data with Nemotron</div>'
            '</div>'
        )

        with gr.Tabs():
            # Tab 1: Full Pipeline
            with gr.Tab("Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        topic_input = gr.Textbox(
                            label="Topic",
                            placeholder="e.g. machine learning, customer support, Python best practices...",
                            lines=1,
                        )
                        domain_input = gr.Dropdown(
                            list(DOMAIN_TEMPLATES.keys()),
                            value="general",
                            label="Domain Template",
                        )
                        count_input = gr.Slider(1, 20, value=5, step=1, label="Number of Pairs")
                        threshold_input = gr.Slider(0.0, 4.0, value=1.5, step=0.1, label="Quality Threshold")
                        seed_input = gr.Textbox(
                            label="Seed Examples (optional JSON)",
                            placeholder='[{"question": "...", "answer": "..."}]',
                            lines=4,
                        )
                        generate_btn = gr.Button("Generate & Score", variant="primary")

                        gr.Markdown("### Quick Examples")
                        ex_btns = [
                            gr.Button("Machine Learning", variant="secondary", size="sm"),
                            gr.Button("Customer Support", variant="secondary", size="sm"),
                            gr.Button("Python Best Practices", variant="secondary", size="sm"),
                            gr.Button("Data Privacy & GDPR", variant="secondary", size="sm"),
                        ]

                    with gr.Column(scale=3):
                        status_output = gr.Textbox(label="Status", interactive=False)
                        stats_output = gr.HTML(label="Summary")
                        pairs_output = gr.HTML(label="Generated Pairs")
                        export_file = gr.File(label="Download JSONL")

                outputs = [status_output, pairs_output, stats_output, export_file]
                generate_btn.click(
                    run_factory,
                    [topic_input, count_input, domain_input, threshold_input, seed_input],
                    outputs,
                )

                ex_topics = ["machine learning", "customer support", "Python best practices", "data privacy and GDPR"]
                ex_domains = ["general", "customer_support", "technical", "legal"]
                for btn, t, d in zip(ex_btns, ex_topics, ex_domains):
                    btn.click(lambda tp=t, dm=d: (tp, dm), outputs=[topic_input, domain_input]).then(
                        run_factory,
                        [topic_input, count_input, domain_input, threshold_input, seed_input],
                        outputs,
                    )

            # Tab 2: Score a single pair
            with gr.Tab("Score a Pair"):
                gr.Markdown("### Score a Single Q&A Pair\nPaste a question and answer to see how the Nemotron Reward model scores it.")
                with gr.Row():
                    with gr.Column():
                        q_input = gr.Textbox(label="Question", lines=3, placeholder="Enter a question...")
                        a_input = gr.Textbox(label="Answer", lines=6, placeholder="Enter the answer to score...")
                        score_btn = gr.Button("Score", variant="primary")
                    with gr.Column():
                        score_output = gr.HTML(label="Scores")
                score_btn.click(score_single, [q_input, a_input], score_output)

            # Tab 3: About
            with gr.Tab("About"):
                gr.Markdown(f"""
### How It Works

**Step 1 — Generate**: The pipeline sends your topic to `{GENERATOR_MODEL}` which generates diverse Q&A pairs.

**Step 2 — Score**: Each pair is evaluated by `{REWARD_MODEL}` across 5 dimensions:
- **Helpfulness** — How useful is the answer?
- **Correctness** — Is the information accurate?
- **Coherence** — Is the answer logically consistent?
- **Complexity** — Is the depth appropriate?
- **Verbosity** — Is the length appropriate?

**Step 3 — Filter**: Pairs below the quality threshold are removed.

**Step 4 — Export**: Passing pairs are exported as JSONL, ready for fine-tuning.

### Domain Templates
{chr(10).join(f'- **{k}**: {v or "No specific guidance"}' for k, v in DOMAIN_TEMPLATES.items())}

### Models Used
- **Generator**: `{GENERATOR_MODEL}`
- **Reward**: `{REWARD_MODEL}`
""")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=PORT, share=False, theme=NVIDIA_THEME, css=CSS)
