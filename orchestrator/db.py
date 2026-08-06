"""Database engine, models, and initialization logic for PostgreSQL (with SQLite fallback)."""

import logging
import uuid
from typing import AsyncGenerator, Dict, Optional
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Integer,
    Boolean,
    Float,
    ForeignKey,
    UniqueConstraint,
    JSON,
    func,
    select,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base, relationship

from orchestrator.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class MCPServer(Base):
    """Registered downstream MCP server metadata."""

    __tablename__ = "mcp_servers"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    transport_type = Column(String(32), nullable=False, default="stdio")
    connection_config = Column(JSON, nullable=False)
    status = Column(String(32), nullable=False, default="active")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    tools = relationship(
        "MCPTool", back_populates="server", cascade="all, delete-orphan"
    )


class MCPTool(Base):
    """Tool definition exposed by a downstream MCP server."""

    __tablename__ = "mcp_tools"

    id = Column(String(128), primary_key=True)  # Format: "server_id:tool_name"
    server_id = Column(
        String(64),
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    input_schema = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    server = relationship("MCPServer", back_populates="tools")

    __table_args__ = (
        UniqueConstraint("server_id", "name", name="unique_server_tool"),
    )


class QueryLog(Base):
    """Log entry for agent queries and returned tool candidates."""

    __tablename__ = "query_logs"

    # UUID string PK set client-side — avoids RETURNING clause incompatibility with aiosqlite.
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(Text, nullable=False)
    discovered_tool_ids = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RetrievalLog(Base):
    """Log entry for vector retrieval and ranking metrics."""

    __tablename__ = "retrieval_logs"

    # UUID string PK set client-side — avoids RETURNING clause incompatibility with aiosqlite.
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_log_id = Column(
        String(36),
        ForeignKey("query_logs.id", ondelete="CASCADE"),
        nullable=False,
    )
    retrieved_tool_ids = Column(JSON, nullable=True)
    scores = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvalRun(Base):
    """Evaluation benchmark run logs."""

    __tablename__ = "eval_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String(255), nullable=False)
    metrics = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ToolInvocationOutcome(Base):
    """Log entry for tool execution success/failure outcomes (Phase 4)."""

    __tablename__ = "tool_invocation_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tool_id = Column(
        String(128),
        ForeignKey("mcp_tools.id", ondelete="CASCADE"),
        nullable=False,
    )
    server_id = Column(String(64), nullable=False)
    tool_name = Column(String(128), nullable=False)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# Active engine and session maker
engine = create_async_engine(settings.POSTGRES_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Create database tables if they do not exist, falling back to SQLite if PostgreSQL is unavailable."""
    global engine, AsyncSessionLocal
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Initialized database schema on '{settings.POSTGRES_URL}'.")
    except Exception as e:
        logger.warning(
            f"Could not connect to PostgreSQL at '{settings.POSTGRES_URL}' ({e}). Falling back to local SQLite database."
        )
        sqlite_url = "sqlite+aiosqlite:///mcp_orchestrator.db"
        engine = create_async_engine(sqlite_url, echo=False, future=True)
        if hasattr(AsyncSessionLocal, "configure"):
            AsyncSessionLocal.configure(bind=engine)
        else:
            AsyncSessionLocal = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Initialized fallback database schema on '{sqlite_url}'.")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency / generator for database session."""
    async with AsyncSessionLocal() as session:
        yield session


async def log_tool_invocation_outcome(
    session: AsyncSession,
    tool_id: str,
    server_id: str,
    tool_name: str,
    success: bool,
    error_message: Optional[str] = None,
    latency_ms: Optional[float] = None,
):
    """Log an invocation outcome record into tool_invocation_outcomes table."""
    record = ToolInvocationOutcome(
        tool_id=tool_id,
        server_id=server_id,
        tool_name=tool_name,
        success=success,
        error_message=error_message,
        latency_ms=latency_ms,
    )
    session.add(record)
    await session.commit()


async def get_past_success_rate(session: AsyncSession, tool_id: str) -> float:
    """Compute historical success rate for a tool. Cold start default = 0.5."""
    stmt_total = select(func.count(ToolInvocationOutcome.id)).where(
        ToolInvocationOutcome.tool_id == tool_id
    )
    stmt_success = select(func.count(ToolInvocationOutcome.id)).where(
        ToolInvocationOutcome.tool_id == tool_id, ToolInvocationOutcome.success == True
    )

    total_res = await session.execute(stmt_total)
    total_count = total_res.scalar() or 0

    if total_count == 0:
        return 0.5  # Neutral cold start default

    success_res = await session.execute(stmt_success)
    success_count = success_res.scalar() or 0

    rate = success_count / float(total_count)
    return round(max(0.0, min(1.0, rate)), 4)
