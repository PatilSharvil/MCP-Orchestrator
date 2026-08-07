"""FastAPI Web Analytics Dashboard for MCP Orchestrator."""

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MCP Orchestrator Analytics Dashboard")

# Serve evaluation chart if present
EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval")
if os.path.exists(EVAL_DIR):
    app.mount("/eval", StaticFiles(directory=EVAL_DIR), name="eval")

# Authoritative experimental metrics from Phase 6 & Phase 7 benchmark
METRICS_DATA = {
    "summary": {
        "token_reduction_pct": 90.3,
        "tokens_saved_per_turn": 6724.9,
        "task_success_rate": 85.0,
        "ranking_latency_ms": 67.16,
    },
    "comparison_table": [
        {
            "mode": "Bypass Baseline (Naive Load-All)",
            "k": 49,
            "tokens": 7450.0,
            "token_reduction": "0.0%",
            "precision": 0.0286,
            "recall": 1.0000,
            "f1": 0.0554,
            "task_success": "90.0%",
            "latency": "2.05 ms",
        },
        {
            "mode": "Full System (Default K=10 Reference)",
            "k": 10,
            "tokens": 1460.5,
            "token_reduction": "80.4%",
            "precision": 0.1250,
            "recall": 0.9250,
            "f1": 0.2174,
            "task_success": "85.0%",
            "latency": "64.51 ms",
        },
        {
            "mode": "Full System (Default K=5 Budget)",
            "k": 5,
            "tokens": 725.1,
            "token_reduction": "90.3%",
            "precision": 0.2100,
            "recall": 0.8000,
            "f1": 0.3262,
            "task_success": "85.0%",
            "latency": "67.16 ms",
        },
        {
            "mode": "Tuned System (w* Calibration K=5)",
            "k": 5,
            "tokens": 725.1,
            "token_reduction": "90.3%",
            "precision": 0.2300,
            "recall": 0.8500,
            "f1": 0.3547,
            "task_success": "85.0%",
            "latency": "67.16 ms",
        },
    ],
    "ablations_k5": [
        {"mode": "Tuned System (w*)", "f1": 0.3547, "precision": 0.2300, "recall": 0.8500},
        {"mode": "No-Ranking (Cosine)", "f1": 0.3714, "precision": 0.2300, "recall": 0.9000},
        {"mode": "Full System (Default)", "f1": 0.3262, "precision": 0.2100, "recall": 0.8000},
        {"mode": "Stage1-Lexical", "f1": 0.3262, "precision": 0.2100, "recall": 0.8000},
        {"mode": "No-ProjMatch", "f1": 0.2976, "precision": 0.1900, "recall": 0.7500},
        {"mode": "No-Shared-Space", "f1": 0.2976, "precision": 0.1900, "recall": 0.7500},
        {"mode": "No-Top-K", "f1": 0.0554, "precision": 0.0286, "recall": 1.0000},
        {"mode": "Bypass Baseline", "f1": 0.0554, "precision": 0.0286, "recall": 1.0000},
    ],
}


@app.get("/api/metrics")
async def get_metrics():
    """Return authoritative benchmark and ablation metrics."""
    return METRICS_DATA


@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    """Serve the single-page HTML analytics dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>MCP Orchestrator Dashboard Index Not Found</h1>"
