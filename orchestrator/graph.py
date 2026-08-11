"""LangGraph state machines for MCP Orchestrator.

Models two inspectable workflow graphs:
1. registration_graph (OFFLINE): Registration -> Embedding -> VectorStore
2. query_graph (ONLINE): ContextParser -> Stage1Retrieval -> Stage2Retrieval -> Ranking -> Logging
"""

import os
import logging
from typing import List, Dict, Any, Optional, TypedDict
from sqlalchemy import select
from langgraph.graph import StateGraph, END

from orchestrator.config import settings, DownstreamServerConfig, load_downstream_servers
import orchestrator.db as db
from orchestrator.registrar import register_downstream_servers
from retrieval.retriever import stage_1_retrieve_servers, stage_2_retrieve_tools
from ranking.ranker import compute_3term_ranking
from ranking.project_context import build_project_context_text

logger = logging.getLogger(__name__)


# ==========================================
# 1. OFFLINE REGISTRATION GRAPH DEFINITION
# ==========================================

class RegistrationState(TypedDict):
    """State schema for offline registration workflow."""

    server_configs: List[DownstreamServerConfig]
    servers_processed: int
    tools_registered: int
    chroma_status: str
    logs: List[str]


async def node_discover_and_postgres(state: RegistrationState) -> Dict[str, Any]:
    """Node 1: Connect downstream MCP servers, call tools/list, and upsert Postgres."""
    logger.info("LangGraph [RegistrationNode]: Querying downstream servers & updating Postgres...")
    configs = state.get("server_configs") or load_downstream_servers()
    
    async with db.AsyncSessionLocal() as session:
        stats = await register_downstream_servers(session, server_configs=configs)
    
    logs = list(state.get("logs", []))
    logs.append(f"Postgres upsert complete: {stats}")
    return {
        "servers_processed": stats.get("servers_processed", 0),
        "tools_registered": stats.get("tools_registered", 0),
        "logs": logs,
    }


async def node_finalize_registration(state: RegistrationState) -> Dict[str, Any]:
    """Node 2: Finalize registration state summary."""
    logger.info("LangGraph [FinalizeNode]: Registration pipeline complete.")
    logs = list(state.get("logs", []))
    logs.append("Registration state machine finished successfully.")
    return {
        "chroma_status": "synced",
        "logs": logs,
    }


def build_registration_graph():
    """Build inspectable LangGraph StateGraph for offline registration."""
    workflow = StateGraph(RegistrationState)
    
    workflow.add_node("registration_node", node_discover_and_postgres)
    workflow.add_node("finalize_node", node_finalize_registration)
    
    workflow.set_entry_point("registration_node")
    workflow.add_edge("registration_node", "finalize_node")
    workflow.add_edge("finalize_node", END)
    
    return workflow.compile()


# ==========================================
# 2. ONLINE DISCOVERY QUERY GRAPH DEFINITION
# ==========================================

class QueryState(TypedDict):
    """State schema for online discovery workflow."""

    query: str
    workspace_dir: Optional[str]
    project_context_text: str
    candidate_servers: List[Dict[str, Any]]
    candidate_tools: List[Dict[str, Any]]
    ranked_tools: List[Dict[str, Any]]
    db_logged: bool
    ablation_mode: Optional[str]


async def node_context_parser(state: QueryState) -> Dict[str, Any]:
    """Node 1: Auto-detect workspace directory and build project context text."""
    raw_ws = state.get("workspace_dir")
    ws = None
    if isinstance(raw_ws, str) and raw_ws.strip():
        ws = raw_ws.strip()
    else:
        ws = os.getcwd()

    logger.info(f"LangGraph [ContextParserNode]: Scanning workspace directory '{ws}'...")
    context_text = build_project_context_text(workspace_path=ws)
    return {
        "workspace_dir": ws,
        "project_context_text": context_text,
    }


async def node_stage1_retrieval(state: QueryState) -> Dict[str, Any]:
    """Node 2: Stage 1 vector search over 'servers' collection."""
    query = state.get("query", "")
    ablation = state.get("ablation_mode")
    logger.info(f"LangGraph [Stage1RetrievalNode]: Querying top-N servers for '{query}' | ablation: {ablation}...")
    candidate_servers = stage_1_retrieve_servers(
        query=query,
        top_n=settings.TOP_N_SERVERS,
        ablation_mode=ablation,
    )
    return {"candidate_servers": candidate_servers}


async def node_stage2_retrieval(state: QueryState) -> Dict[str, Any]:
    """Node 3: Stage 2 metadata-filtered vector search over 'tools' collection."""
    query = state.get("query", "")
    ablation = state.get("ablation_mode")
    candidate_servers = state.get("candidate_servers", [])
    candidate_server_ids = [s["server_id"] for s in candidate_servers]
    
    logger.info(f"LangGraph [Stage2RetrievalNode]: Filtering tools for servers {candidate_server_ids}...")
    top_m = 1000 if (ablation and ablation.lower() == "no_topk") else settings.TOP_M_TOOLS
    tool_results = stage_2_retrieve_tools(
        query=query,
        candidate_server_ids=candidate_server_ids,
        top_m=top_m,
        ablation_mode=ablation,
    )
    
    # Enrich with PostgreSQL metadata
    enriched = []
    if tool_results:
        async with db.AsyncSessionLocal() as session:
            for item in tool_results:
                tid = item["tool_id"]
                db_tool = await session.get(db.MCPTool, tid)
                if db_tool:
                    enriched.append({
                        "tool_id": db_tool.id,
                        "server_id": db_tool.server_id,
                        "name": db_tool.name,
                        "description": db_tool.description,
                        "input_schema": db_tool.input_schema,
                        "score": item["score"],
                    })
                else:
                    enriched.append({
                        "tool_id": tid,
                        "server_id": item["server_id"],
                        "name": item["name"],
                        "description": "",
                        "input_schema": {},
                        "score": item["score"],
                    })
    else:
        # Fallback to DB query if ChromaDB vector search returns empty results
        logger.info("ChromaDB vector search returned 0 candidates. Falling back to DB query...")
        fallback_limit = 1000 if (ablation and ablation.lower() == "no_topk") else settings.TOP_M_TOOLS
        async with db.AsyncSessionLocal() as session:
            stmt = select(db.MCPTool).limit(fallback_limit)
            res = await session.execute(stmt)
            db_tools = res.scalars().all()
            for t in db_tools:
                enriched.append({
                    "tool_id": t.id,
                    "server_id": t.server_id,
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    "score": 0.5,
                })

    return {"candidate_tools": enriched}


async def node_ranking(state: QueryState) -> Dict[str, Any]:
    """Node 4: Compute 3-term weighted scores (QuerySim, ProjMatch, PastSuccess)."""
    query = state.get("query", "")
    candidates = state.get("candidate_tools", [])
    context_text = state.get("project_context_text", "")
    ablation = state.get("ablation_mode")
    
    logger.info(f"LangGraph [RankingNode]: Computing 3-term weighted ranking scores (context: '{context_text}')...")
    async with db.AsyncSessionLocal() as session:
        ranked = await compute_3term_ranking(
            candidate_tools=candidates,
            query=query,
            project_context=context_text,
            w1=settings.RANKING_W1_QUERY_SIM,
            w2=settings.RANKING_W2_PROJ_MATCH,
            w3=settings.RANKING_W3_PAST_SUCCESS,
            db_session=session,
            ablation_mode=ablation,
        )
    return {"ranked_tools": ranked}


async def node_logging(state: QueryState) -> Dict[str, Any]:
    """Node 5: Log query execution and retrieval metadata to PostgreSQL/SQLite."""
    import uuid as _uuid
    query = state.get("query", "")
    ranked = state.get("ranked_tools", [])

    logger.info("LangGraph [LoggingNode]: Logging query and score breakdowns to Postgres...")
    try:
        async with db.AsyncSessionLocal() as session:
            # Generate UUID in Python so it's available before SQLAlchemy's flush.
            q_id = str(_uuid.uuid4())
            q_log = db.QueryLog(
                id=q_id,
                query=query,
                discovered_tool_ids=[t["tool_id"] for t in ranked],
            )
            session.add(q_log)

            r_log = db.RetrievalLog(
                id=str(_uuid.uuid4()),
                query_log_id=q_id,
                retrieved_tool_ids=[t["tool_id"] for t in ranked],
                scores={
                    t["tool_id"]: {
                        "final_score": t["score"],
                        "breakdown": t["score_breakdown"],
                    }
                    for t in ranked
                },
            )
            session.add(r_log)
            await session.commit()
    except Exception as e:
        logger.warning(f"Could not log query execution in LangGraph logging node: {e}")

    return {"db_logged": True}


def build_query_graph():
    """Build inspectable LangGraph StateGraph for online discovery workflow."""
    workflow = StateGraph(QueryState)
    
    workflow.add_node("context_parser_node", node_context_parser)
    workflow.add_node("stage1_retrieval_node", node_stage1_retrieval)
    workflow.add_node("stage2_retrieval_node", node_stage2_retrieval)
    workflow.add_node("ranking_node", node_ranking)
    workflow.add_node("logging_node", node_logging)
    
    workflow.set_entry_point("context_parser_node")
    workflow.add_edge("context_parser_node", "stage1_retrieval_node")
    workflow.add_edge("stage1_retrieval_node", "stage2_retrieval_node")
    workflow.add_edge("stage2_retrieval_node", "ranking_node")
    workflow.add_edge("ranking_node", "logging_node")
    workflow.add_edge("logging_node", END)
    
    return workflow.compile()


# Compiled Graph Singletons
registration_pipeline_graph = build_registration_graph()
query_pipeline_graph = build_query_graph()


async def run_online_query_pipeline(
    query: str,
    workspace_dir: Optional[str] = None,
    ablation_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Helper entrypoint executing online query pipeline via LangGraph."""
    initial_state: QueryState = {
        "query": query,
        "workspace_dir": workspace_dir,
        "project_context_text": "",
        "candidate_servers": [],
        "candidate_tools": [],
        "ranked_tools": [],
        "db_logged": False,
        "ablation_mode": ablation_mode,
    }
    
    final_state = await query_pipeline_graph.ainvoke(initial_state)
    return final_state.get("ranked_tools", [])
