"""Phase 6 Benchmark Runner Script.

Executes 20 benchmark tasks across Mode A (Bypass Baseline) and Mode B (Full Orchestrator Pipeline),
computes comprehensive metrics (Context & Token Reduction, Precision/Recall/F1@K, Task Success Rate,
Recovery Rate, Latency Breakdown, and Estimated Cost Savings), and generates a GFM Markdown comparison table.

Task Success Rate definition:
  - Mode A: success_check tool is always in context (all 49 tools visible). Task succeeds if invoke_tool
    call succeeds (isError=False).
  - Mode B: Task succeeds ONLY IF (a) success_check.tool_id appears in the top-K ranked candidates
    AND (b) invoke_tool call succeeds. If the tool isn't in the top-K, it can't be used — task fails.

Recovery Rate definition:
  - Mode A: same as Task Success Rate (no ranking, no recovery concept — all tools visible).
  - Mode B: if primary invocation succeeds → Recovery=1. If primary fails or tool not in top-K →
    try fallback_tool_ids that ARE in the top-K candidates. Recovery=1 if any fallback succeeds,
    else Recovery=0.
"""

import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Tuple
import tiktoken

from orchestrator.db import init_db, AsyncSessionLocal
from retrieval.embedder import sync_all_embeddings_from_db
from orchestrator.server import discover_tools, invoke_tool

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Illustrative Prompt Token Pricing Assumption ($3.00 per 1,000,000 prompt tokens)
COST_PER_MILLION_TOKENS_ASSUMPTION = 3.00


def get_token_encoder():
    """Get tiktoken cl100k_base encoder with character fallback."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or len(text)//4 fallback."""
    if not text:
        return 0
    enc = get_token_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text) // 4


def compute_schema_tokens(tools: List[Dict[str, Any]]) -> int:
    """Calculate total prompt token count for a list of candidate tool JSON schemas."""
    total = 0
    for t in tools:
        schema_str = json.dumps({
            "name": t.get("name"),
            "description": t.get("description"),
            "input_schema": t.get("input_schema"),
        })
        total += count_tokens(schema_str)
    return total


def compute_precision_recall_f1(returned_tool_ids: List[str], ground_truth_tool_ids: List[str]) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F1 score @ K."""
    if not returned_tool_ids or not ground_truth_tool_ids:
        return 0.0, 0.0, 0.0

    returned_set = set(returned_tool_ids)
    gt_set = set(ground_truth_tool_ids)

    tp = len(returned_set.intersection(gt_set))
    precision = tp / float(len(returned_set))
    recall = tp / float(len(gt_set))

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def _fmt_delta_pct(a: float, b: float) -> str:
    """Format a percentage delta (b - a) with sign."""
    d = round(b - a, 2)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


def _fmt_delta_ms(a: float, b: float) -> str:
    """Format a millisecond delta (b - a) with sign."""
    d = round(b - a, 2)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f} ms"


async def run_benchmark_task_suite(tasks_filepath: str = "eval/benchmark_tasks.json") -> Dict[str, Any]:
    """Execute all benchmark tasks against Mode A (Bypass Baseline) and Mode B (Orchestrator Pipeline)."""
    await init_db()

    # Sync embeddings into ChromaDB vector store
    async with AsyncSessionLocal() as session:
        await sync_all_embeddings_from_db(session)

    if not os.path.exists(tasks_filepath):
        raise FileNotFoundError(f"Benchmark tasks fixture file '{tasks_filepath}' not found.")

    with open(tasks_filepath, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"\n==========================================================================")
    print(f"   STARTING MCP ORCHESTRATOR BENCHMARK HARNESS ({len(tasks)} TASKS)   ")
    print(f"==========================================================================\n")

    # Accumulators
    bypass_metrics = {
        "tokens": [], "precision": [], "recall": [], "f1": [],
        "stage1_hits": [], "stage2_hits": [],
        "task_success": [], "recovery_success": [],
        "ranking_latency_ms": [], "spawn_latency_ms": [], "total_latency_ms": [],
    }
    orc_metrics = {
        "tokens": [], "precision": [], "recall": [], "f1": [],
        "stage1_hits": [], "stage2_hits": [],
        "task_success": [], "recovery_success": [],
        "ranking_latency_ms": [], "spawn_latency_ms": [], "total_latency_ms": [],
    }

    # Per-task breakdown rows
    per_task_rows = []

    for task in tasks:
        tid = task["task_id"]
        domain = task["domain"]
        query = task["query"]
        gt_tools = task.get("ground_truth_tool_ids", [])
        exp_servers = task.get("expected_server_ids", [])
        succ_check = task.get("success_check", {})
        fallbacks = task.get("fallback_tool_ids", [])

        # Build a full tool_id for the primary success-check tool
        primary_full_id = (
            f"{succ_check['server_id']}:{succ_check['tool_id']}"
            if succ_check else ""
        )

        # ---------------------------------------------------------
        # 1. MODE A: Bypass Baseline (returns ALL tools, all visible)
        # ---------------------------------------------------------
        t_start_bypass = time.perf_counter()
        tools_bypass = await discover_tools(query=query, bypass=True)
        ranking_latency_bypass = round((time.perf_counter() - t_start_bypass) * 1000, 2)

        bypass_tids = [t["tool_id"] for t in tools_bypass]
        bypass_sids = list(set(t["server_id"] for t in tools_bypass))
        tokens_bypass = compute_schema_tokens(tools_bypass)

        p_b, r_b, f1_b = compute_precision_recall_f1(bypass_tids, gt_tools)
        s1_hit_b = 1.0 if any(s in bypass_sids for s in exp_servers) else 0.0
        s2_hit_b = 1.0 if any(t in bypass_tids for t in gt_tools) else 0.0

        # Mode A: primary tool is ALWAYS in context (bypass returns all 49).
        # Task success = invoke_tool call succeeds.
        spawn_lat_b = 0.0
        task_succ_b = 0.0
        rec_succ_b = 0.0
        bypass_fail_reason = "no success_check defined"

        if succ_check:
            t_exec_start = time.perf_counter()
            inv_res = await invoke_tool(
                server_id=succ_check["server_id"],
                tool_id=succ_check["tool_id"],
                arguments=succ_check.get("arguments", {}),
            )
            spawn_lat_b = round((time.perf_counter() - t_exec_start) * 1000, 2)
            is_err = inv_res.get("isError", False)
            task_succ_b = 0.0 if is_err else 1.0
            # Mode A recovery = same as task success (all tools visible, no fallback needed)
            rec_succ_b = task_succ_b
            if is_err:
                bypass_fail_reason = "invoke_tool returned isError=True"
            else:
                bypass_fail_reason = "ok"

        bypass_metrics["tokens"].append(tokens_bypass)
        bypass_metrics["precision"].append(p_b)
        bypass_metrics["recall"].append(r_b)
        bypass_metrics["f1"].append(f1_b)
        bypass_metrics["stage1_hits"].append(s1_hit_b)
        bypass_metrics["stage2_hits"].append(s2_hit_b)
        bypass_metrics["task_success"].append(task_succ_b)
        bypass_metrics["recovery_success"].append(rec_succ_b)
        bypass_metrics["ranking_latency_ms"].append(ranking_latency_bypass)
        bypass_metrics["spawn_latency_ms"].append(spawn_lat_b)
        bypass_metrics["total_latency_ms"].append(ranking_latency_bypass + spawn_lat_b)

        # ---------------------------------------------------------
        # 2. MODE B: Full Orchestrator Pipeline (3-Term Ranked top-K)
        # ---------------------------------------------------------
        t_start_orc = time.perf_counter()
        tools_orc = await discover_tools(query=query, workspace_dir=".", bypass=False)
        ranking_latency_orc = round((time.perf_counter() - t_start_orc) * 1000, 2)

        orc_tids = [t["tool_id"] for t in tools_orc]
        orc_sids = list(set(t["server_id"] for t in tools_orc))
        tokens_orc = compute_schema_tokens(tools_orc)

        p_o, r_o, f1_o = compute_precision_recall_f1(orc_tids, gt_tools)
        s1_hit_o = 1.0 if any(s in orc_sids for s in exp_servers) else 0.0
        s2_hit_o = 1.0 if any(t in orc_tids for t in gt_tools) else 0.0

        spawn_lat_o = 0.0
        task_succ_o = 0.0
        rec_succ_o = 0.0
        orc_fail_reason = "no success_check defined"
        primary_in_context = False

        if succ_check:
            # Step 1: Is the primary success-check tool visible in Mode B's top-K?
            primary_in_context = primary_full_id in orc_tids

            if not primary_in_context:
                # Tool not in context → agent cannot call it → task fails
                task_succ_o = 0.0
                orc_fail_reason = f"primary tool '{primary_full_id}' not in top-K candidates"

                # Recovery: try fallbacks that ARE in top-K
                fallbacks_in_context = [f for f in fallbacks if f in orc_tids]
                if fallbacks_in_context:
                    t_exec_start = time.perf_counter()
                    fb_server_id = fallbacks_in_context[0].split(":")[0]
                    fb_tool_name = fallbacks_in_context[0].split(":", 1)[1]
                    fb_res = await invoke_tool(
                        server_id=fb_server_id,
                        tool_id=fb_tool_name,
                        arguments=succ_check.get("arguments", {}),
                    )
                    spawn_lat_o = round((time.perf_counter() - t_exec_start) * 1000, 2)
                    if not fb_res.get("isError", False):
                        rec_succ_o = 1.0
                        orc_fail_reason += f" — recovered via '{fallbacks_in_context[0]}'"
                    else:
                        rec_succ_o = 0.0
                        orc_fail_reason += " — recovery also failed"
                else:
                    rec_succ_o = 0.0
                    orc_fail_reason += " — no fallback in top-K either"
            else:
                # Primary is in context → invoke it
                t_exec_start = time.perf_counter()
                inv_res = await invoke_tool(
                    server_id=succ_check["server_id"],
                    tool_id=succ_check["tool_id"],
                    arguments=succ_check.get("arguments", {}),
                )
                spawn_lat_o = round((time.perf_counter() - t_exec_start) * 1000, 2)
                is_err = inv_res.get("isError", False)
                task_succ_o = 0.0 if is_err else 1.0

                if task_succ_o == 1.0:
                    rec_succ_o = 1.0
                    orc_fail_reason = "ok"
                else:
                    # Primary invocation failed — try fallbacks in top-K
                    orc_fail_reason = "invoke_tool returned isError=True"
                    fallbacks_in_context = [f for f in fallbacks if f in orc_tids]
                    for fb_full_id in fallbacks_in_context:
                        fb_server_id = fb_full_id.split(":")[0]
                        fb_tool_name = fb_full_id.split(":", 1)[1]
                        fb_res = await invoke_tool(
                            server_id=fb_server_id,
                            tool_id=fb_tool_name,
                            arguments=succ_check.get("arguments", {}),
                        )
                        if not fb_res.get("isError", False):
                            rec_succ_o = 1.0
                            orc_fail_reason += f" — recovered via '{fb_full_id}'"
                            break

        orc_metrics["tokens"].append(tokens_orc)
        orc_metrics["precision"].append(p_o)
        orc_metrics["recall"].append(r_o)
        orc_metrics["f1"].append(f1_o)
        orc_metrics["stage1_hits"].append(s1_hit_o)
        orc_metrics["stage2_hits"].append(s2_hit_o)
        orc_metrics["task_success"].append(task_succ_o)
        orc_metrics["recovery_success"].append(rec_succ_o)
        orc_metrics["ranking_latency_ms"].append(ranking_latency_orc)
        orc_metrics["spawn_latency_ms"].append(spawn_lat_o)
        orc_metrics["total_latency_ms"].append(ranking_latency_orc + spawn_lat_o)

        status_b = "PASS" if task_succ_b == 1.0 else "FAIL"
        status_o = "PASS" if task_succ_o == 1.0 else "FAIL"
        rec_str = "PASS" if rec_succ_o == 1.0 else "FAIL"
        in_ctx  = "Y" if primary_in_context else "N"
        per_task_rows.append({
            "task_id": tid,
            "domain": domain,
            "primary_in_ctx_b": True,   # always true for bypass
            "primary_in_ctx_o": primary_in_context,
            "task_succ_b": task_succ_b,
            "task_succ_o": task_succ_o,
            "rec_o": rec_succ_o,
            "bypass_reason": bypass_fail_reason,
            "orc_reason": orc_fail_reason,
        })

        print(
            f"Task [{tid}] ({domain:<12}): "
            f"ModeA={status_b}({bypass_fail_reason[:30]}) | "
            f"ModeB={status_o}(InCtx={in_ctx}, {orc_fail_reason[:35]}) | "
            f"Rec={rec_str} | Tokens A={tokens_bypass} B={tokens_orc}"
        )

    # ---------------------------------------------------------
    # AGGREGATE SUMMARY METRIC COMPUTATIONS
    # ---------------------------------------------------------
    num_t = float(len(tasks))
    avg_tokens_b = sum(bypass_metrics["tokens"]) / num_t
    avg_tokens_o = sum(orc_metrics["tokens"]) / num_t

    token_reduction_pct = round(((avg_tokens_b - avg_tokens_o) / avg_tokens_b) * 100.0, 2) if avg_tokens_b > 0 else 0.0
    context_reduction_pct = token_reduction_pct
    total_tokens_saved = sum(bypass_metrics["tokens"]) - sum(orc_metrics["tokens"])
    estimated_cost_savings = round((total_tokens_saved / 1_000_000.0) * COST_PER_MILLION_TOKENS_ASSUMPTION, 4)

    avg_prec_b  = round(sum(bypass_metrics["precision"]) / num_t, 4)
    avg_prec_o  = round(sum(orc_metrics["precision"]) / num_t, 4)
    avg_rec_b   = round(sum(bypass_metrics["recall"]) / num_t, 4)
    avg_rec_o   = round(sum(orc_metrics["recall"]) / num_t, 4)
    avg_f1_b    = round(sum(bypass_metrics["f1"]) / num_t, 4)
    avg_f1_o    = round(sum(orc_metrics["f1"]) / num_t, 4)

    s1_hit_rate_b = round(sum(bypass_metrics["stage1_hits"]) / num_t * 100, 2)
    s1_hit_rate_o = round(sum(orc_metrics["stage1_hits"]) / num_t * 100, 2)
    s2_hit_rate_b = round(sum(bypass_metrics["stage2_hits"]) / num_t * 100, 2)
    s2_hit_rate_o = round(sum(orc_metrics["stage2_hits"]) / num_t * 100, 2)

    task_succ_rate_b = round(sum(bypass_metrics["task_success"]) / num_t * 100, 2)
    task_succ_rate_o = round(sum(orc_metrics["task_success"]) / num_t * 100, 2)
    rec_succ_rate_b  = round(sum(bypass_metrics["recovery_success"]) / num_t * 100, 2)
    rec_succ_rate_o  = round(sum(orc_metrics["recovery_success"]) / num_t * 100, 2)

    avg_rank_lat_b  = round(sum(bypass_metrics["ranking_latency_ms"]) / num_t, 2)
    avg_rank_lat_o  = round(sum(orc_metrics["ranking_latency_ms"]) / num_t, 2)
    avg_spawn_lat_b = round(sum(bypass_metrics["spawn_latency_ms"]) / num_t, 2)
    avg_spawn_lat_o = round(sum(orc_metrics["spawn_latency_ms"]) / num_t, 2)
    avg_total_lat_b = round(sum(bypass_metrics["total_latency_ms"]) / num_t, 2)
    avg_total_lat_o = round(sum(orc_metrics["total_latency_ms"]) / num_t, 2)

    report_data = {
        "num_tasks": int(num_t),
        "pricing_assumption": f"${COST_PER_MILLION_TOKENS_ASSUMPTION:.2f} per 1M prompt tokens (ILLUSTRATIVE)",
        "per_task_breakdown": per_task_rows,
        "mode_a_bypass": {
            "avg_schema_tokens": round(avg_tokens_b, 1),
            "precision_at_k": avg_prec_b, "recall_at_k": avg_rec_b, "f1_at_k": avg_f1_b,
            "stage1_hit_rate_pct": s1_hit_rate_b, "stage2_hit_rate_pct": s2_hit_rate_b,
            "task_success_rate_pct": task_succ_rate_b, "recovery_rate_pct": rec_succ_rate_b,
            "avg_ranking_overhead_ms": avg_rank_lat_b,
            "avg_subprocess_spawn_ms": avg_spawn_lat_b,
            "avg_total_latency_ms": avg_total_lat_b,
        },
        "mode_b_orchestrator": {
            "avg_schema_tokens": round(avg_tokens_o, 1),
            "precision_at_k": avg_prec_o, "recall_at_k": avg_rec_o, "f1_at_k": avg_f1_o,
            "stage1_hit_rate_pct": s1_hit_rate_o, "stage2_hit_rate_pct": s2_hit_rate_o,
            "task_success_rate_pct": task_succ_rate_o, "recovery_rate_pct": rec_succ_rate_o,
            "avg_ranking_overhead_ms": avg_rank_lat_o,
            "avg_subprocess_spawn_ms": avg_spawn_lat_o,
            "avg_total_latency_ms": avg_total_lat_o,
        },
        "improvements": {
            "token_reduction_pct": token_reduction_pct,
            "context_reduction_pct": context_reduction_pct,
            "total_tokens_saved": total_tokens_saved,
            "estimated_cost_savings_usd": estimated_cost_savings,
        },
    }

    # ---------------------------------------------------------
    # Per-task breakdown table
    # ---------------------------------------------------------
    per_task_md = "\n### Per-Task Success / Failure Breakdown\n\n"
    per_task_md += "| Task ID | Domain | In Ctx A/B | Succ A | Succ B | Rec B | Notes (Mode B) |\n"
    per_task_md += "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n"
    for row in per_task_rows:
        ctx_a  = "Y" if row["primary_in_ctx_b"] else "N"
        ctx_b  = "Y" if row["primary_in_ctx_o"] else "N"
        succ_a = "PASS" if row["task_succ_b"] == 1.0 else "FAIL"
        succ_b = "PASS" if row["task_succ_o"] == 1.0 else "FAIL"
        rec_b  = "PASS" if row["rec_o"] == 1.0 else "FAIL"
        note = row["orc_reason"][:60]
        per_task_md += f"| `{row['task_id']}` | {row['domain']} | {ctx_a} / {ctx_b} | {succ_a} | {succ_b} | {rec_b} | {note} |\n"

    # ---------------------------------------------------------
    # Summary comparison table (corrected delta column)
    # ---------------------------------------------------------
    # Recall delta: lower Mode B recall than Mode A is a trade-off for precision
    recall_delta = round(avg_rec_o - avg_rec_b, 4)
    recall_sign = "+" if recall_delta >= 0 else ""

    md_table = f"""
# MCP Orchestrator — Phase 6 Benchmark Evaluation Report

* **Total Tasks Evaluated**: {int(num_t)} tasks (spanning Filesystem, Memory Graph, GitHub, & Cross-Domain)
* **Pricing Model Assumption**: ${COST_PER_MILLION_TOKENS_ASSUMPTION:.2f} / 1M prompt tokens *(Illustrative assumption — adjust per actual model/provider used)*

### Benchmark Results Comparison Table

| Metric | Mode A: Bypass Baseline (Naive Load-All) | Mode B: MCP Orchestrator (3-Term Ranked) | Improvement / Delta |
| :--- | :---: | :---: | :---: |
| **Exposed Schema Tokens (Avg/Query)** | `{avg_tokens_b:.1f} tokens` | `{avg_tokens_o:.1f} tokens` | **-{token_reduction_pct}% Token Reduction** |
| **Context Window Reduction (%)** | `0.0%` (baseline) | `{context_reduction_pct}%` | **{context_reduction_pct}% Context Reduction** |
| **Precision @ K** | `{avg_prec_b:.4f}` | `{avg_prec_o:.4f}` | **+{(avg_prec_o - avg_prec_b):.4f} Precision** |
| **Recall @ K** | `{avg_rec_b:.4f}` | `{avg_rec_o:.4f}` | `{recall_sign}{recall_delta:.4f} Recall` |
| **F1 Score @ K** | `{avg_f1_b:.4f}` | `{avg_f1_o:.4f}` | **+{(avg_f1_o - avg_f1_b):.4f} F1 Score** |
| **Stage 1 Server Hit Rate (%)** | `{s1_hit_rate_b:.1f}%` | `{s1_hit_rate_o:.1f}%` | `{_fmt_delta_pct(s1_hit_rate_b, s1_hit_rate_o)}` |
| **Stage 2 Tool Hit Rate (%)** | `{s2_hit_rate_b:.1f}%` | `{s2_hit_rate_o:.1f}%` | `{_fmt_delta_pct(s2_hit_rate_b, s2_hit_rate_o)}` |
| **Task Success Rate (%)** | `{task_succ_rate_b:.1f}%` | `{task_succ_rate_o:.1f}%` | `{_fmt_delta_pct(task_succ_rate_b, task_succ_rate_o)}` |
| **Recovery Rate (%)** | `{rec_succ_rate_b:.1f}%` | `{rec_succ_rate_o:.1f}%` | `{_fmt_delta_pct(rec_succ_rate_b, rec_succ_rate_o)}` |
| **Python Ranking Overhead (Avg)** | `{avg_rank_lat_b:.2f} ms` | `{avg_rank_lat_o:.2f} ms` | `{_fmt_delta_ms(avg_rank_lat_b, avg_rank_lat_o)}` |
| **Subprocess Spawn Overhead (stdio)** | `{avg_spawn_lat_b:.2f} ms` | `{avg_spawn_lat_o:.2f} ms` | *(npx child process startup cost — both modes share same downstream servers)* |
| **Total System Latency (Avg)** | `{avg_total_lat_b:.2f} ms` | `{avg_total_lat_o:.2f} ms` | `{_fmt_delta_ms(avg_total_lat_b, avg_total_lat_o)}` |
| **Estimated Cost Savings ($)** | `$0.00` | **${estimated_cost_savings:.4f} USD** | *(Saved {total_tokens_saved:,} tokens)* |
"""

    print(md_table)
    print(per_task_md)
    return report_data


if __name__ == "__main__":
    asyncio.run(run_benchmark_task_suite())
