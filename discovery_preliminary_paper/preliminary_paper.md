# Intelligent Dynamic Tool Retrieval, Ranking, and Stateful Execution Middleware for Autonomous AI Agents (MCP Orchestrator) — Preliminary Manuscript

> **Scope note:** This is a simplified, preliminary conceptual paper prepared for a college paper-presentation event. It describes the problem area, design intent, and general architecture of the MCP Orchestrator project at a high level. It intentionally omits detailed implementation logic, ranking formulas, system configurations, experimental datasets, benchmark results, and ablation analysis, which are maintained separately in the project's full research records. No quantitative performance claims are made in this manuscript.

## Title

Intelligent Dynamic Tool Retrieval, Ranking, and Stateful Execution Middleware for Autonomous AI Agents (MCP Orchestrator)

## Abstract

Autonomous AI agents increasingly rely on external tools exposed through the Model Context Protocol (MCP) to access files, development platforms, databases, and other services. In conventional integrations, an agent loads the interface descriptions of all available tools into its context before beginning a task. As the number of connected servers grows, this upfront loading consumes a large share of the limited context window, increases operating cost, and confronts the agent with many irrelevant choices at decision time.

This preliminary paper introduces MCP Orchestrator, a middleware gateway placed between AI agents and downstream MCP servers. Instead of exposing every tool description at once, the gateway exposes a small number of general-purpose operations: one for discovering tools relevant to the current request, and one for requesting execution of a selected tool through the gateway. Discovery combines semantic matching of the user request against tool descriptions, alignment with the general characteristics of the active workspace, and consideration of prior execution history. Execution is proxied to the appropriate downstream server, and basic outcome information is recorded to support auditability and future ranking decisions.

We outline the motivation, objectives, general architecture, and basic methodology in conceptual terms, discuss expected benefits and limitations, and identify directions for future work. No experimental evaluation is reported here; evaluation design and results are outside the scope of this preliminary manuscript.

## Keywords

AI agents; Model Context Protocol; tool retrieval; tool ranking; middleware; context management; stateful execution

## 1. Introduction

Large language model based agents extend their capabilities by invoking external tools. A coding assistant, for example, may need to read project files, query issue trackers, search documentation, or interact with databases. The Model Context Protocol (MCP) provides a shared standard for exposing such capabilities: independent server processes describe their tools in a structured format, and agent clients can discover and call them over standard transports.

A straightforward way to use MCP is for the agent to fetch the full list of tool descriptions from every connected server at session start and keep them in context throughout the interaction. This approach is simple to implement and works well at small scale. It becomes problematic as deployments grow from one or two servers to a fleet covering many domains. Each additional server adds interface descriptions that must be carried in the prompt on subsequent turns, leaving less room for user instructions, retrieved code, tool outputs, and conversation history. It also confronts the underlying language model with a large decision space in which many tools are irrelevant to the immediate task.

MCP Orchestrator explores an alternative organization: a gateway that consolidates downstream servers behind a narrow, stable interface. Agents interact with the gateway rather than with each server directly. When a task appears to require tool use, the agent first asks the gateway for a short list of candidate tools relevant to the current request. It then requests execution of a chosen candidate, and the gateway forwards the call to the responsible server and returns the result. The gateway maintains metadata about registered servers and tools, an index to support relevance search, and records of past invocations.

This manuscript presents that concept at an introductory level. Section 2 states the problem, Section 3 lists objectives for this preliminary stage, Section 4 situates the work among established research areas, Sections 5–7 describe the proposed system and methodology in general terms, and Sections 8–12 discuss anticipated benefits, open issues, limitations, conclusions, and future scope.

## 2. Problem Statement

The central problem addressed here is the cost of exposing too many tool descriptions to an agent at once in MCP-based workflows.

First, context capacity is finite. Language models accept a bounded amount of input per request. Tool interface descriptions, while individually modest in size, accumulate across servers. In multi-turn sessions, this material is typically re-supplied alongside conversation history, so overhead compounds across turns and reduces the space available for task content such as source files, error traces, and prior tool outputs.

Second, large choice sets complicate tool selection. When dozens of tools with overlapping names or purposes are presented together, the agent must discriminate among them from descriptions alone. Established studies of long-context behavior suggest that relevant information placed among substantial irrelevant material can be harder for models to use reliably. In practical terms, this can manifest as selection of a plausible but suboptimal tool, incorrect arguments, or unnecessary clarification turns.

Third, static upfront exposure is insensitive to task context. The same full tool set is shown whether the user asks a conversational question, requests a filesystem operation, or works in a particular programming environment. A task-aware approach would instead surface a small, task-relevant subset and defer details of remaining tools until they are plausibly needed.

Fourth, direct fan-out to many servers complicates administration. Each new server requires client-side configuration, and there is no central point to record which tools were suggested or invoked, or whether invocations succeeded. This limits observability for developers and evaluators.

The problem, therefore, is not that tools exist, but that the prevailing method of presenting them does not scale gracefully with the number of servers, the length of sessions, or the diversity of tasks.

## 3. Objectives

Within the limited scope of this preliminary manuscript, the objectives are conceptual and architectural rather than empirical:

1. **Describe a gateway pattern** that sits between MCP-compatible agents and a fleet of downstream MCP servers, presenting a small stable interface upstream while managing fan-out downstream.
2. **Outline just-in-time discovery**, in which candidate tools are retrieved per request based on broad relevance signals rather than preloaded exhaustively.
3. **Outline relevance-based ranking in general terms**, considering the content of the user request, general characteristics of the active workspace, and summary information from prior executions, without specifying exact scoring procedures.
4. **Outline stateful proxied execution**, in which the gateway forwards selected calls to the appropriate server and records basic invocation outcomes for transparency and later analysis.
5. **Preserve extensibility and compatibility**, so that adding or removing a downstream server does not require changes to agent-side code, and communication continues to use standard MCP transports.
6. **Deliberately defer empirical claims.** Defining evaluation tasks, collecting measurements, and comparing alternatives are left to the separate, comprehensive phase of the work and are not attempted here.

## 4. Brief Literature Review

This section identifies established research areas relevant to the proposed system. It does not survey every related effort and does not report comparative results. Full bibliographic entries with verification status are provided in the accompanying `references.md` file.

**Tool use by language models.** Early work demonstrated that language models can be trained or prompted to interleave text generation with calls to external utilities such as calculators, search interfaces, and question-answering systems. This established the pattern of treating tool invocation as part of the generation process rather than as a separate hard-coded step.

**Large API collections and retrieval.** As the number of available interfaces grew from a handful to hundreds or thousands, subsequent work highlighted the difficulty of selecting the correct interface from descriptions alone. Proposed responses included retrieving a subset of candidate interfaces with a separate retrieval step before generation, and organizing retrieval hierarchically, for example by first identifying a relevant service category and then a specific operation within it. These ideas motivate the two-level view adopted here: narrow the field to relevant servers before shortlisting individual tools.

**Reasoning and acting in agents.** Another line of work studies agents that alternate between producing intermediate reasoning and taking actions through available interfaces, using observations from prior actions to inform subsequent decisions. This perspective informs the gateway's emphasis on per-request discovery followed by observed execution, rather than single-shot upfront planning.

**Sentence representations for semantic search.** Dense text embeddings derived from transformer models enable similarity search over short descriptions. By representing both a user request and tool descriptions in a shared representation space, a system can shortlist candidates whose purposes are semantically close to the request, even when wording differs. Standard techniques for deriving such representations underpin the semantic component of discovery described below.

**Behavior of models over long contexts.** Controlled studies of question answering and retrieval over long inputs have reported that model performance is sensitive to the amount and placement of irrelevant material. While specific findings depend on model and task, the general implication is widely cited as motivation for keeping prompts focused: supplying fewer, better-targeted tool descriptions is preferable to supplying all descriptions indiscriminately.

**Protocol standardization.** The Model Context Protocol provides the interoperability foundation for this work. It defines how servers describe tools and how clients list and invoke them over local and network transports. The gateway builds on this standard rather than replacing it: it acts as a server from the agent's perspective and as a client from the perspective of downstream servers.

Together, these areas suggest that a middleware layer combining retrieval, ranking, and proxied execution is a plausible architectural response to tool overload, while leaving open the detailed design and evaluation choices that distinguish specific systems.

## 5. Proposed System

MCP Orchestrator is proposed as a gateway middleware with two responsibilities: helping agents find relevant tools when needed, and carrying out approved tool calls on their behalf.

From the agent's perspective, the gateway looks like an ordinary MCP server offering a very small set of general operations. One operation accepts a natural-language description of the current need and returns a short list of candidate tools drawn from across the connected fleet. A second operation accepts the identity of a chosen tool along with its arguments, forwards the call to the server that owns the tool, and returns the outcome. The agent therefore never needs to hold descriptions of the full fleet in context; it sees only the gateway interface plus the shortlisted candidates for the task at hand.

From the perspective of downstream servers, the gateway acts as a managing client. It reads a declarative registry of servers, establishes connections over the appropriate transports for local and remote servers, collects tool descriptions, and maintains them in a searchable index. When servers are added, removed, or updated, the gateway refreshes its metadata and index without requiring changes to agent configurations.

Three broad information sources inform discovery. The first is the semantic relationship between the request and tool descriptions: tools whose documented purposes resemble the request are preferred. The second is the general character of the active workspace, such as the kinds of files and declared dependencies present, which provides weak contextual evidence about which classes of tools are plausible. The third is summarized execution history: tools with a record of repeated failures in similar circumstances should be treated cautiously, while consistently usable tools need no special penalty. This manuscript describes these signals qualitatively; how they are represented, combined, or weighted is part of the detailed design and is intentionally not specified here.

Execution through the gateway is stateful in a limited sense: the gateway records which candidates were suggested for which requests and whether proxied invocations succeeded or failed, along with basic timing information. These records support inspection and debugging and provide input to future ranking decisions. They are not presented here as a learning algorithm or adaptive policy, only as an audit trail with potential reuse.

## 6. General System Architecture

At a conceptual level, the system comprises four groups of elements. Figure-like description is given in text, as no implementation diagram is included in this preliminary version.

**1. Agent layer.** Any MCP-compatible agent or development environment. Agents connect to the gateway using standard transports and are unchanged except that they address the gateway rather than individual servers. Examples of the class of clients include command-line coding assistants, desktop assistants, and editor extensions that support MCP.

**2. Gateway layer.** The central middleware. Functionally, it contains:
- *A registry and indexing subsystem* that tracks downstream servers, collects their tool descriptions, and maintains a searchable index over those descriptions.
- *A discovery subsystem* that, given a request and optional workspace location, shortlists candidate servers, then shortlists candidate tools within those servers, and orders the shortlist by broad relevance.
- *An execution router* that validates a requested tool identity, forwards the call with the supplied arguments to the owning server, and returns the server's response to the agent.
- *A persistence subsystem* with two roles: a metadata and history store for server records, tool records, and invocation logs; and a vector index supporting semantic similarity search over server and tool descriptions.

**3. Downstream server fleet.** The set of independent MCP servers providing domain capabilities, such as filesystem access, version control hosting, documentation search, memory, or databases. Servers may run as local subprocesses or as remote endpoints. Each retains ownership of its tools; the gateway does not reimplement domain logic.

**4. Supporting storage.** A relational store for structured metadata and logs, and a vector index for retrieval. The design favors components that can run in a lightweight local configuration for individual developers and in a more robust shared configuration for teams, with graceful degradation if an optional component is unavailable. Specific products, versions, and deployment topologies are implementation choices outside the scope of this manuscript.

Interaction proceeds in two phases. In the discovery phase, the agent submits a request description; the gateway consults workspace context where available, searches the index, ranks candidates, and returns a bounded shortlist. In the execution phase, the agent selects a candidate and submits arguments; the gateway routes the call, observes the outcome, records it, and relays the result. Conversational requests with no apparent tool need may yield an empty shortlist, in which case the agent proceeds without tool use.

## 7. Basic Methodology

The working methodology is described here as a sequence of stages. Each stage is stated functionally; concrete algorithms, parameter values, and thresholds are omitted as they belong to the detailed phase of the work.

**Stage A — Registration and indexing.** Administrators declare downstream servers in a registry. The gateway connects to each server, retrieves tool listings, normalizes descriptions, and populates the metadata store and search index. Changes to the registry trigger re-indexing without requiring agent-side modifications.

**Stage B — Request-time context preparation.** Upon receiving a discovery request, the gateway notes the request text and, when a workspace location is supplied, inspects its general characteristics in a read-only manner, such as prevalent file types and declared dependencies. The purpose is to form a lightweight contextual background, not to analyze file contents in depth.

**Stage C — Two-level candidate retrieval.** Retrieval proceeds from coarse to fine. First, servers whose overall purpose appears related to the request are shortlisted, which constrains the subsequent search. Second, tools belonging to those servers are shortlisted by similarity between the request and their descriptions. Restricting the tool search to previously shortlisted servers is intended to reduce cross-domain false matches. A lexical fallback is available in principle if semantic search cannot be performed, but its design is not elaborated here.

**Stage D — Relevance ordering.** Shortlisted candidates are ordered by an overall notion of relevance that considers request similarity, workspace fit, and prior execution summaries. Candidates falling below a general notion of minimal relevance may be withheld so that unrelated requests do not trigger spurious tool suggestions. The ordering is presented to the agent with brief explanatory metadata, such as the owning server and a short description, so that the agent can make an informed selection.

**Stage E — Proxied execution and logging.** The agent invokes the execution operation with a chosen server identity, tool identity, and arguments. The gateway validates the reference, forwards the call, awaits the downstream response, records the outcome and basic timing, and returns the response. Failures in downstream servers are reported transparently rather than hidden, and the log entry preserves what was attempted for later inspection.

No datasets, metrics, or results are associated with these stages in this manuscript. How to measure retrieval quality, task completion, resource usage, and latency, and how to ablate individual components, are evaluation questions reserved for the comprehensive study.

## 8. Expected Benefits

The following benefits are anticipated on architectural grounds. They are stated as expectations to be tested in future evaluation, not as measured findings.

- **Reduced prompt burden.** By replacing exhaustive upfront exposure with per-request shortlists, the amount of tool-related material held in the agent's context is expected to be substantially smaller, leaving more capacity for task content and dialogue history.
- **More focused decisions.** Presenting a small number of task-relevant candidates is expected to simplify the agent's selection problem compared with choosing among the full fleet.
- **Context sensitivity.** Taking the request and general workspace characteristics into account is expected to favor tools that fit the immediate situation over superficially similar alternatives from unrelated domains.
- **Operational transparency.** Centralizing discovery and execution records is expected to make it easier to answer basic questions such as which tools were suggested, which were invoked, and whether invocations succeeded.
- **Extensibility.** Encapsulating downstream diversity behind a stable gateway interface is expected to lower the cost of adding or retiring servers, since agent configurations address the gateway rather than each server individually.
- **Standard compatibility.** Because the gateway speaks MCP in both directions, existing compliant agents and servers are expected to interoperate without protocol modifications.

Whether and to what degree these expectations hold will depend on detailed design choices and empirical conditions that are not examined here.

## 9. Preliminary Discussion

Several design tensions are worth noting even at this early stage.

The first concerns shortlist size. Returning too few candidates risks omitting the tool the agent needs, forcing an additional discovery round; returning too many reintroduces the overload the gateway is meant to alleviate. The appropriate size likely depends on task diversity and on how clearly requests discriminate among tools, and it should be treated as an evaluation question rather than fixed prematurely.

The second concerns reliance on descriptions. Retrieval quality is bounded by the quality of tool documentation. Terse, vague, or overlapping descriptions will produce ambiguous shortlists regardless of retrieval machinery. This suggests that documentation hygiene in downstream servers is a first-order concern for deployments.

The third concerns history reuse. Prior execution outcomes are informative but potentially misleading if conditions change, for example when credentials are corrected or a server is upgraded. Any use of history should therefore distinguish persistent unreliability from transient failures, and should default to neutral treatment when evidence is sparse. The present manuscript takes no position on how this balance should be implemented.

The fourth concerns trust and privilege. Consolidating execution behind a gateway centralizes responsibility for forwarding calls, including calls with side effects. This makes access control, logging, and administrator oversight more important, not less. A gateway should not be understood as a security boundary by itself; it must be deployed alongside appropriate authentication, authorization, and review practices for the underlying servers.

Finally, the gateway introduces an additional network or process hop between agent and tool. This indirection is the price of consolidation. Keeping its overhead modest and its failure modes transparent — for example, by reporting downstream errors faithfully — is essential if the gateway is to be a practical intermediary rather than a bottleneck.

## 10. Limitations

This preliminary manuscript and the conceptual system it describes have clear limitations:

- **No empirical validation is presented.** All statements about benefits are expectations. No tasks were run, no measurements were taken, and no comparisons were performed for this manuscript.
- **Design detail is intentionally withheld.** Scoring procedures, combination strategies, index construction, storage schemas, configuration formats, and operational parameters are not specified and cannot be reconstructed from this text.
- **Scope is restricted to description-level discovery.** The gateway shortlists tools based on documented purposes and general context; it does not verify that a tool will succeed for a specific argument set, nor does it synthesize or repair arguments.
- **Dependence on upstream and downstream quality.** The approach assumes reasonably documented tools and a capable underlying language model. It does not compensate for fundamentally poor descriptions or for model limitations in reasoning about tool use.
- **Single-gateway assumptions.** Issues of scaling to very large fleets, replicating the gateway for availability, isolating tenants, and enforcing fine-grained permissions are acknowledged but not addressed here.
- **Literature coverage is illustrative.** The brief review highlights relevant directions without claiming completeness or novelty with respect to any specific prior system.

These limitations are consistent with the purpose of a preliminary paper: to articulate an idea and its organization clearly enough for discussion, while deferring substantiation to later work.

## 11. Conclusion

This manuscript has presented MCP Orchestrator in simplified, preliminary form: a middleware gateway that replaces static preloading of all MCP tool descriptions with per-request discovery of a short candidate list and proxied, logged execution of the selected tool. The general architecture separates agent-facing stability from downstream diversity, and the basic methodology proceeds from registration and indexing through context-aware retrieval, relevance ordering, and stateful execution.

No claims of measured improvement, efficiency, or superiority are made. The contribution intended here is organizational and expository: to state the problem of tool overload in accessible terms, to describe a plausible gateway-shaped response using only broad design concepts, and to set out what would need to be evaluated before stronger conclusions could be drawn. Detailed design, implementation, and evaluation remain the subject of the project's separate comprehensive records.

## 12. Future Scope

Several directions are identified for subsequent investigation, none of which is claimed as present capability:

- **Evaluation methodology.** Construction of representative task suites spanning multiple server domains, with definitions of retrieval quality, task completion, resource usage, and latency suitable for comparing gateway-assisted and direct-integration approaches.
- **Ranking and retrieval studies.** Systematic examination of how different relevance signals and retrieval organizations affect shortlist quality, including sensitivity to description quality and workspace context.
- **Robustness and failure handling.** Clearer treatment of transient versus persistent failures, retry disciplines, graceful degradation when optional stores are unavailable, and faithful propagation of downstream errors.
- **Scale and operations.** Exploration of indexing and deployment strategies for larger fleets, replication for availability, multi-user audit requirements, and administrative tooling for registry management.
- **Safety and governance.** Integration with authentication, authorization, and review workflows for side-effecting tools, and study of how centralization affects least-privilege practices.
- **Agent interaction patterns.** Observation of how agents use shortlists in practice, including when they request re-discovery, how they choose among candidates, and how shortlist presentation affects their decisions.

Each direction presupposes the detailed and empirical work explicitly excluded from this preliminary manuscript.

## References

Brief citations are listed here for readability. Complete entries with verification status are given in `references.md`.

[1] Schick et al., Toolformer, 2023.
[2] Patil et al., Gorilla, 2023.
[3] Yao et al., ReAct, 2022/2023.
[4] Reimers and Gurevych, Sentence-BERT, 2019.
[5] Liu et al., Lost in the Middle, 2023/2024.
[6] Anthropic, Model Context Protocol documentation/specification.

---

*Repository basis:* This description was prepared from a high-level inspection of the project's public structure (gateway server, downstream client management, retrieval and ranking components, relational and vector persistence, and dashboard/evaluation scaffolding) without reproducing internal logic, parameters, or experimental records. All technical statements above are intentionally kept at a conceptual level consistent with that inspection.
