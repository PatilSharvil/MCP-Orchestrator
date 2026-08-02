"""Database engine, models, and initialization logic for PostgreSQL."""

import datetime
from typing import AsyncGenerator
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    BigInteger,
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

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    discovered_tool_ids = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RetrievalLog(Base):
    """Log entry for vector retrieval and ranking metrics."""

    __tablename__ = "retrieval_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    query_log_id = Column(
        BigInteger,
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

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_name = Column(String(255), nullable=False)
    metrics = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# Async SQLAlchemy Engine & Session Maker
engine = create_async_engine(settings.POSTGRES_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Create PostgreSQL tables if they do not exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency / generator for database session."""
    async with AsyncSessionLocal() as session:
        yield session
