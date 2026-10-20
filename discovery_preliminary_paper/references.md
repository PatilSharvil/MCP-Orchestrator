# References — MCP Orchestrator Preliminary Manuscript

This file lists the references cited in `preliminary_paper.md`, with verification status as of 2026-10-09. Only works whose bibliographic details could be verified through primary sources (publisher, arXiv, or official documentation site) are included. No additional citations are claimed.

## Verification method

- arXiv entries were verified by fetching `arxiv.org/abs/<id>` and confirming title, authors, date, and identifier.
- The Sentence-BERT entry was additionally verified against the ACL Anthology (EMNLP-IJCNLP 2019, D19-1410).
- The Lost in the Middle entry was verified via web search results pointing to arXiv and the Transactions of the Association for Computational Linguistics (TACL).
- The Model Context Protocol entry was verified by fetching `https://modelcontextprotocol.io/` (official documentation site, a living specification).

## References

### [1] Verified

Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., and Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv preprint arXiv:2302.04761 [cs.CL]. Submitted 9 Feb 2023. DOI: 10.48550/arXiv.2302.04761.

- Status: **Verified** (arXiv abstract page fetched; title, authors, date, and arXiv ID confirmed).
- Relevance: Early demonstration of language models deciding when and how to call external tools via APIs.

### [2] Verified

Patil, S. G., Zhang, T., Wang, X., and Gonzalez, J. E. (2023). Gorilla: Large Language Model Connected with Massive APIs. arXiv preprint arXiv:2305.15334 [cs.CL, cs.AI]. Submitted 24 May 2023. DOI: 10.48550/arXiv.2305.15334.

- Status: **Verified** (arXiv abstract page fetched; title, authors, date, and arXiv ID confirmed).
- Relevance: Identifies accuracy and hallucination challenges when selecting among many APIs; motivates retrieval-assisted tool selection.

### [3] Verified

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv preprint arXiv:2210.03629 [cs.CL]. Published as a conference paper at ICLR 2023.

- Status: **Verified** (arXiv ID 2210.03629 confirmed via search citing arXiv abstract and PDF; authors and title confirmed).
- Relevance: Paradigm interleaving reasoning traces with actions through external interfaces; informs per-request discovery followed by observed execution.

### [4] Verified

Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3982–3992. Association for Computational Linguistics. DOI: 10.18653/v1/D19-1410. Also arXiv:1908.10084 [cs.CL].

- Status: **Verified** (ACL Anthology D19-1410 and arXiv:1908.10084 confirmed via search/fetch).
- Relevance: Foundation for dense sentence representations compared with cosine similarity, as used broadly in semantic search over tool descriptions.

### [5] Verified

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. arXiv preprint arXiv:2307.03172 [cs.CL]. Published in Transactions of the Association for Computational Linguistics (TACL), vol. 12, pp. 157–173, 2024. DOI: 10.1162/tacl_a_00638.

- Status: **Verified** (arXiv ID and TACL publication confirmed via search results from arXiv, ACL Anthology, and MIT Press).
- Relevance: Empirical analysis of degraded use of long input contexts when relevant information is surrounded by irrelevant material; motivates focused shortlists over exhaustive preloading.

### [6] Verified (living specification)

Anthropic. Model Context Protocol (MCP) — Official Documentation and Specification. Available at https://modelcontextprotocol.io/. Initially released November 2024; content updated as a living specification.

- Status: **Verified** (official site fetched on 2026-10-09; site identifies MCP as an open standard for connecting AI applications to external systems via clients and servers).
- Relevance: Defines the open protocol for tool description, discovery (`tools/list`-style listing), and invocation on which the gateway builds. Cite as a website/specification rather than a fixed paper; readers should consult the version current at time of reading.

## Excluded / not cited

The following were considered but are **not** cited in the preliminary manuscript because independent verification was not completed during this task, or because they would imply detailed evaluation claims outside the manuscript's scope:

- ToolBench / ToolLLM line of work — relevant to tool-learning evaluation methodology, but omitted to avoid unverified version details.
- AnyTool and related hierarchical retrieval systems — omitted for the same verification reason.
- Project-internal benchmark reports, configuration files, and experimental logs — intentionally excluded per task instructions; they belong to the separate comprehensive research records.

## Integrity statement

To the best of the preparer's checks on 2026-10-09:

- No experimental results, numerical improvements, performance measurements, or claims of superiority are asserted in the manuscript.
- No citations are fabricated. All six entries above were confirmed against primary sources as noted.
- Where the literature is broad, the manuscript states that coverage is illustrative rather than exhaustive.
- Detailed ranking formulas, weights, thresholds, system configurations, datasets, and benchmark tables from the full research were intentionally omitted and are not reproduced here.
