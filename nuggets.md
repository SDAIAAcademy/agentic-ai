Audience: Programmers comfortable with Python (functions, classes, type hints, async, JSON, HTTP, git, the terminal) who have used ChatGPT but cannot yet say what an LLM is, how it differs from ChatGPT, or how to call one from code.

# Knowledge nuggets

How this file was built. Pass 1 walked the original decks slide by slide, starting from the audience line alone. For every
slide it asked: what must the reader already know to follow this? Anything not yet in this file was either explained by
rewriting the slide or by adding a background slide, and the slide's own new knowledge was then added here. Pass 2 read
the dependency arrows (`⟵`) and reordered the course so that every nugget is taught before it is used. Numbers follow the
final order of the course. `1.2` means Day 1, Session 2; `1.0` is the Day 1 opening. A nugget may be mentioned earlier
only as a labelled preview ("tool calling, Session 3").

## Day 1 · From a language model to an agent

**Opening (1.0)**

- **N1** · 1.0 · This week you build an *agent system*: software in which a language model chooses steps and calls your code, plus the engineering that makes it safe, measurable and shippable.
- **N2** · 1.0 · Case study: Madar Municipal Services (fictional) issues five permit types (BLD building, REN renovation, SGN signage, FTR food truck, EVT event). Its assistant Rafeeq answers citizens in Arabic and English. All data is synthetic: 10 policies, a SQLite database, a 30-email inbox. "Today" in the labs is 1 November 2026.
- **N3** · 1.0 · *Prompt injection* (first definition): text from an outside source (an email, a web page, a document) written to look like instructions for the AI. Two emails in Madar's inbox carry one. Full treatment on Day 4. ⟵ N2
- **N4** · 1.0 · PDPL, Saudi Arabia's Personal Data Protection Law: personal data (national IDs, mobile numbers) is collected and kept only as needed, and disclosed only to the right person. That is why Rafeeq verifies identity first. ⟵ N2
- **N5** · 1.0 · Labs: Python notebooks on Colab or locally with `uv`; one free OpenRouter key in `.env` or Colab Secrets; the course helper module `agentkit` imported as `K`; identical requests are replayed from `labs/cache`.

**Session 1 · What an LLM is, and how to call one (1.1)**

- **N6** · 1.1 · A *language model* is a function: given a sequence of tokens, it returns a probability for every possible next token.
- **N7** · 1.1 · *Tokens* are pieces of text (whole words, word parts, punctuation) from a fixed vocabulary of roughly 100k–250k entries. The model reads and writes token IDs. English averages about ¾ of a word per token; Arabic usually needs more tokens per word (the "Arabic token tax"). ⟵ N6
- **N8** · 1.1 · *Generation* repeats one step: predict the next-token distribution, pick a token, append it, run again, until a stop token or a length limit. Every output token is one more model pass, so output is slower and pricier than input. ⟵ N6, N7
- **N9** · 1.1 · *Sampling*: picking from the distribution with some randomness (temperature). The same input can give different outputs, so LLM behaviour is *stochastic*; you measure it over several runs. ⟵ N8
- **N10** · 1.1 · *Large* language model (LLM): billions of learned numbers (*weights*). *Pre-training* on huge text teaches next-token prediction; *post-training* (instruction tuning, human feedback) teaches it to follow instructions and to chat. ⟵ N6
- **N11** · 1.1 · *Open-weight* models (Gemma, Qwen, Llama) can be downloaded and hosted by anyone; *closed* models (GPT, Claude, Gemini) are reachable only through their vendor's API. ⟵ N10
- **N12** · 1.1 · ChatGPT is a *product*: an app with a UI, saved chats, memory, web search and file handling, built around OpenAI's GPT models. An LLM is the model alone. Same split: the Claude app vs Claude models, the Gemini app vs Gemini models. From code you call the model through an API and build the rest yourself. ⟵ N10
- **N13** · 1.1 · *Chat format*: a conversation is a list of messages with roles: `system` (the developer's standing instructions, the *system prompt*), `user`, `assistant`. A *chat template* flattens the list into one token sequence with markers, and the model continues it as the assistant. ⟵ N7, N8
- **N14** · 1.1 · The model is *stateless*: it remembers nothing between calls. Every call resends the whole conversation; the app (or your code) keeps the history. ⟵ N13
- **N15** · 1.1 · *Context window*: the maximum tokens per call, input plus output (for example 128k to 1M). Everything the model knows about the task must be in it. ⟵ N7, N14
- **N16** · 1.1 · A model knows only its training data, up to a *knowledge cutoff*, and writes fluent text even when it does not know: *hallucination*. Facts about your data must be put in the context or fetched with tools. ⟵ N10
- **N17** · 1.1 · An *LLM API* is an HTTPS endpoint: you POST JSON (model, messages, parameters) with an API key and get JSON back. You pay per token, with input and output priced separately (per million tokens). ⟵ N7, N13
- **N18** · 1.1 · Who serves models: *model makers* (OpenAI, Anthropic, Google), *inference engines and providers* that host open weights (vLLM, SGLang, Ollama, LM Studio, cloud providers), and *gateways* that forward to many providers behind one key (OpenRouter, Hugging Face Inference Providers). ⟵ N11, N17
- **N19** · 1.1 · *Chat Completions* (`/v1/chat/completions`, OpenAI, 2023) became the common API: almost every server accepts it, so the `openai` Python SDK reaches any of them by changing `base_url`. ⟵ N17, N18
- **N20** · 1.1 · A response tells you `model` (who served it), `finish_reason` (`stop`, `length`, `tool_calls`) and `usage` (prompt, completion and cached tokens; cost on OpenRouter). `max_tokens` caps output; `length` means the answer was cut. ⟵ N8, N19
- **N21** · 1.1 · *Streaming* sends tokens as they are generated. *Time to first token* drives perceived speed; *tokens per second* drives total time. ⟵ N8
- **N22** · 1.1 · Free models on OpenRouter end in `:free`: 50 requests a day (1,000 after a one-time 10-credit purchase) and 20 a minute, shared by everyone, so expect HTTP 429 (rate limited). Defences: retry with backoff, a fallback list (`models`), a response cache. Course default: `google/gemma-4-31b-it:free`. ⟵ N18, N19
- **N23** · 1.1 · Preview: *tool calling* is a model's trained ability to answer with a structured request to run a function you described, instead of text. Model lists mark it as `tools`. Taught in 1.3.
- **N24** · 1.1 · Public benchmarks: Artificial Analysis (quality, speed, price per provider), LMArena (human votes on blind chats), BFCL (function-calling correctness), τ²-bench (tool-using agents following a policy with a simulated customer, reported as pass^k), Terminal-Bench (terminal tasks, model and harness scored together). Shortlist with them; decide with your own evaluation (3.4). Caveats: *contamination* (test items leaked into training data), different harnesses, mostly English. ⟵ N23

**Session 2 · Structured outputs and the Responses API (1.2)**

- **N25** · 1.2 · Code needs data, not prose: triaging an email means one label from a closed set, a permit ID, a language, a yes/no. ⟵ N2
- **N26** · 1.2 · *JSON Schema* describes the allowed shape of JSON (types, enums, required fields). Pydantic builds one from a Python class and validates JSON against it.
- **N27** · 1.2 · *Structured outputs*: send a schema; in *strict* mode the server constrains generation so that only tokens keeping the output valid can be picked. Some models and servers treat the schema as a hint, so: request strict where supported, validate always, repair once (send the error back), then fail loudly. ⟵ N8, N26
- **N28** · 1.2 · *Decisions*: guards, routers and judges are typed answers, not prose. Three question types: predicate (yes/no), choice (one of a set), score (a level on a scale); several independent questions per call. A probability the model *writes* is just tokens, not a measurement; OpenAI's Decisions API (beta, paid) reads real probabilities from the token distribution. ⟵ N6, N27
- **N29** · 1.2 · Three API shapes run the industry: Chat Completions (messages with roles), OpenAI *Responses* (`/v1/responses`, 2025, typed *items*) and Anthropic *Messages* (`/v1/messages`, messages of typed *content blocks*). Most engines and gateways serve all three; the *Open Responses* specification (January 2026) made the Responses shape portable. ⟵ N19
- **N30** · 1.2 · Responses: `instructions` is the system prompt as a parameter (resend it each call); `input`; the output is a list of items (`message`, `reasoning`, `function_call`, …); `output_text` joins the text; `text.format` carries a schema; responses are stored unless `store=False`. ⟵ N13, N29
- **N31** · 1.2 · *Reasoning models* generate hidden thinking tokens before the answer. They help with multi-step problems, cost output tokens, and come back as reasoning items or thinking blocks (a summary, or encrypted content to pass back). ⟵ N8, N30
- **N32** · 1.2 · Responses streams *semantic events* (`response.output_item.added`, `…delta`, `response.completed`), so a UI can react to each item as it starts. ⟵ N21, N30
- **N33** · 1.2 · Conversation state: *stateless replay* (`store=False`, you resend the items; works on every server; you can trim and audit) vs *server-side state* (`previous_response_id`, Conversations, background runs, server compaction: OpenAI-only, less to send, more lock-in). The old Assistants API shut down in August 2026. ⟵ N14, N30
- **N34** · 1.2 · The portable core to build on: items, function tools, `text.format`, `store=False`. ⟵ N29, N33

**Session 3 · Tools and the Messages API (1.3)**

- **N35** · 1.3 · A *tool call*: the model answers with a function name and JSON arguments instead of text; your code runs the function and sends the result back; the model continues. The model never executes anything itself. Tools fix N16: fresh facts and real actions. ⟵ N16, N23
- **N36** · 1.3 · A *tool definition* is a name, a description and a JSON Schema for the arguments. It is sent with every request (it costs tokens) and is all the model sees of your function. ⟵ N26, N35
- **N37** · 1.3 · Rafeeq's tools are typed, documented Python functions: type hints become the schema, the docstring becomes the descriptions. *Read* tools (search_policies, get_permit, compute_fee, list_slots) and *write* tools (book/reschedule_inspection, create_refund, escalate, send_email). Business rules live in the tool code and fail with a `ToolError` whose message tells the model what to do. ⟵ N2, N36
- **N38** · 1.3 · Messages API: `max_tokens` is required; `system` is a top-level parameter; `content` is a list of blocks (`text`, `tool_use`, `thinking`); `stop_reason` is `end_turn`, `tool_use`, `max_tokens`, `pause_turn` or `refusal`. The `anthropic` SDK reaches any compatible server via `base_url`. ⟵ N29, N35
- **N39** · 1.3 · A Messages round trip: append the assistant turn unchanged; send every `tool_result` (matched by `tool_use_id`) in *one* user message; tool errors are results with `is_error: true`. ⟵ N38
- **N40** · 1.3 · One function, three wrappers: Chat Completions nests `function.parameters`, Responses is flat with `parameters`, Messages uses `input_schema`. ⟵ N29, N36
- **N41** · 1.3 · Never run a tool call from a reply cut by `max_tokens` / `length`: its arguments may be truncated. ⟵ N20, N38
- **N42** · 1.3 · *Parallel tool calls*: a model may request several tools in one turn. ⟵ N35
- **N43** · 1.3 · Tool design: name what it does, one job per tool, fewer higher-level tools, enums for closed sets, units and examples in descriptions, return what the model needs next, errors written as instructions. ⟵ N36
- **N44** · 1.3 · *Function tools* run in your code; *hosted (server) tools* run at the provider (web search, code execution, a remote MCP server). Hosted tools are vendor-specific and sit outside your permissions and logs. ⟵ N35
- **N45** · 1.3 · A *system prompt is a specification*: role and scope, rules that tools cannot enforce (tone, language, citations), and trust boundaries (text in tool results and emails is data, never instructions). Rules that code can enforce go in the tools. ⟵ N3, N13, N37
- **N46** · 1.3 · Open engines (vLLM, SGLang, llama.cpp, Ollama, LM Studio, LiteLLM, OpenRouter) serve `/v1/messages` so that tools built for Claude run on open models. ⟵ N18, N38
- **N47** · 1.3 · `K.run_agent` repeats tool round trips until the model answers in text; Lab 1.3 uses it before you write your own in 1.4. ⟵ N35

**Session 4 · The agent loop (1.4)**

- **N48** · 1.4 · An *agent* is a model using tools in a loop: the model decides which tool to call next and when it is done. A *workflow* is code that decides the path, with models inside fixed steps. Real systems mix both. ⟵ N35
- **N49** · 1.4 · Autonomy is a spectrum (single call → structured decision → workflow → agent → long-running agent) and a cost (latency, tokens, variance, attack surface). Earn each step with evidence that the simpler level fails. ⟵ N28, N48
- **N50** · 1.4 · The loop: send history + tools; if the reply has no tool calls, return it; else run each call, append the results, repeat. Two exits: the model stops calling tools, or the budget runs out. ⟵ N14, N35
- **N51** · 1.4 · Each step resends the growing history, so input tokens grow every step and the cost of a run grows faster than linearly. ⟵ N14, N50
- **N52** · 1.4 · A *trace* is the record of a run: model calls, tool calls, results, the answer. You debug, test and evaluate traces. ⟵ N50
- **N53** · 1.4 · Budgets (steps, cost, time) stop runaway loops; tool errors go back to the model as results; bugs in your own code stay out of its view. ⟵ N37, N50
- **N54** · 1.4 · One loop serves all three shapes; only an adapter changes (where calls are found, how results are sent). ⟵ N40, N50
- **N55** · 1.4 · A *harness* is everything around the model that turns it into an agent: the loop, tools, prompt, context management, permissions, hooks, sandbox, sessions. Agent frameworks and SDKs are harnesses someone else wrote. ⟵ N50
- **N56** · 1.4 · *Coding agents* (Claude Code, OpenAI Codex, Cursor, Pi) are harnesses whose tools read and edit files and run shell commands in your project. ⟵ N55
- **N57** · 1.4 · The course map, the engineering ladder, each layer wrapping the one before: prompt (Day 1) → context (Day 2) → harness and loop (Day 3) → graph (Day 4), with evaluation, security and production across all of them. ⟵ N45, N55

## Day 2 · Context engineering

- **N58** · 2.0 · *Hooks*: functions the loop calls at fixed points. `K.run_agent` offers `after_tool` (rewrite a result before the model sees it), `permission` and `approve` (Day 3). ⟵ N50
- **N59** · 2.0 · *Context engineering*: choosing what enters the window at each step (Karpathy, 2025). ⟵ N15
- **N60** · 2.1 · Agents read far more than they write: Manus (an agent product) reported about 100 input tokens per output token. ⟵ N51
- **N61** · 2.1 · *Context rot*: accuracy falls as input grows, even when everything fits (Chroma, 18 models, 2025). The effective context is smaller than the window. ⟵ N15
- **N62** · 2.1 · Four context failures: poisoning, distraction, confusion, clash. ⟵ N61
- **N63** · 2.1 · Four strategies: write (outside the window), select (only what this step needs), compress, isolate (a separate window). ⟵ N62
- **N64** · 2.1 · *Prompt caching*: the provider keeps the model's internal computation for a prompt prefix for a few minutes; a later request with the same prefix is read, not recomputed, and billed at a fraction. Matching starts at the first token: one changed token invalidates everything after it. Anthropic marks breakpoints with `cache_control`; OpenAI caches automatically (`prompt_cache_key`). ⟵ N8, N17
- **N65** · 2.1 · Shrinking history: trim tool results (an `after_tool` hook), compact (summary + recent turns), clear (replace old results with a placeholder: context editing); both APIs offer server-side versions. ⟵ N58, N63
- **N66** · 2.1 · A summary is a new copy of personal data: data minimisation (PDPL) applies to context too. ⟵ N4, N65
- **N67** · 2.2 · *RAG* (retrieval-augmented generation): find relevant passages (*chunks* of documents) and put them in the prompt so the model answers from them. ⟵ N16
- **N68** · 2.2 · Pipeline RAG (always retrieve once, with the user's words) vs retrieval as a tool (the model decides whether and what to search, and may search again). ⟵ N35, N67
- **N69** · 2.2 · *BM25*: keyword ranking; a chunk scores higher for containing the query's terms, rare terms weigh more, long chunks are normalised. Exact IDs win; it needs shared words. ⟵ N67
- **N70** · 2.2 · *Embeddings*: a model turns text into a vector of numbers so that similar meanings land close together (cosine similarity). A multilingual model puts Arabic and English in one space. `potion-multilingual-128M` (model2vec) runs on a CPU. ⟵ N67
- **N71** · 2.2 · Arabic normalisation for keyword search: remove diacritics and tatweel; unify alef forms, taa marbuta and alef maqsura. ⟵ N69
- **N72** · 2.2 · *Hybrid search* fuses rankings with reciprocal rank fusion: score = Σ 1/(60 + rank). ⟵ N69, N70
- **N73** · 2.2 · A retrieval eval: questions with gold document IDs; *recall@k* = share of questions whose gold document is in the top k. ⟵ N67
- **N74** · 2.2 · *Agentic search* (list, grep, read tools) wins on structured, exact, fresh corpora; loses on huge or fuzzy ones. When the corpus fits, a cached prompt with all of it can beat any retriever. ⟵ N64, N68
- **N75** · 2.2 · Citations: every passage carries its ID; the answer cites it; an eval can check the claim against the cited source. ⟵ N67
- **N76** · 2.3 · Three kinds of context that outlive a call: session memory (the history), long-term memory (a store the agent reads and writes), standing context (files the harness loads every time). ⟵ N14, N55
- **N77** · 2.3 · Memory is an attack surface: an injected instruction written to memory returns in later sessions. ⟵ N3, N76
- **N78** · 2.3 · The memory tool (Anthropic, `memory_20250818`): commands the model calls (`view`, `create`, `str_replace`, `insert`, `delete`, `rename`) on `/memories/…`; your code owns the storage, refuses paths outside the root, caps sizes and says what must never be stored. ⟵ N36, N76
- **N79** · 2.3 · `AGENTS.md`: an open Markdown format for standing context at a project's root, read by most coding agents; loaded into every conversation, so short. ⟵ N56, N76
- **N80** · 2.3 · *Agent Skills* (`SKILL.md`: front matter with name and description, then the procedure; optional scripts): only the index is always in context, the body loads when a task matches (*progressive disclosure*). Open standard (agentskills.io). Skills from public hubs are code you run. ⟵ N63, N76
- **N81** · 2.3 · Just-in-time context: for each piece ask "every call, sometimes, or only now?". ⟵ N63, N78, N79, N80
- **N82** · 2.4 · *JSON-RPC 2.0*: a request is `{"jsonrpc", "id", "method", "params"}`, the reply carries the same `id` with `result` or `error`.
- **N83** · 2.4 · *MCP* (Model Context Protocol, Anthropic 2024, now under the Agentic AI Foundation of the Linux Foundation) standardises how any agent host discovers and calls tools from any server: M + N integrations instead of M × N. ⟵ N36, N82
- **N84** · 2.4 · Host (the app with the model), client (one connection per server), server (offers capabilities); transports: *stdio* (a local subprocess) and *Streamable HTTP* (remote). ⟵ N83
- **N85** · 2.4 · Server primitives: tools (the model decides), resources (data by URI; the host or user decides), prompts (templates; the user decides). Clients can offer sampling, elicitation and roots back. ⟵ N84
- **N86** · 2.4 · *PEP 723*: a single Python file declares its dependencies in a comment block; `uv run file.py` installs them in a throwaway environment. ⟵ N5
- **N87** · 2.4 · MCP Python SDK v2: `MCPServer` (was `FastMCP`); type hints become schemas; only `ToolError` messages reach the model. ⟵ N37, N85, N86
- **N88** · 2.4 · Spec revision 2026-07-28 is stateless: no handshake or session; each request carries version and capabilities in `_meta`; method and tool name also in headers; `server/discover`. ⟵ N82, N84
- **N89** · 2.4 · Three ways a model reaches a server: your own host (you see every call), a harness's config file, or a model API's hosted MCP tool. ⟵ N44, N55, N84
- **N90** · 2.4 · Supply-chain risks: tool poisoning (instructions in a description enter context when tools are listed), rug pulls (tools change after approval), over-broad credentials. ⟵ N3, N36, N83

## Day 3 · Harness and loop engineering

- **N91** · 3.1 · Agent = model + harness: LangChain (a company that builds agent frameworks) raised its Terminal-Bench score from 52.8 to 66.5 by changing only the harness. ⟵ N24, N55
- **N92** · 3.1 · *Guides* steer before an action (prompts, tool design, skills, permissions); *sensors* check after it (tests, linters, tool errors, evals, logs). ⟵ N45, N91
- **N93** · 3.1 · Pi: a minimal open-source coding-agent harness: four tools, a short sectioned prompt, sessions as a JSONL tree, JSON/RPC/SDK interfaces; no per-call approval and no sandbox. ⟵ N56
- **N94** · 3.1 · Pi's loop is N50 with more exits and an event stream. ⟵ N50, N93
- **N95** · 3.1 · A *CVE* is the public ID of a disclosed security vulnerability.
- **N96** · 3.1 · *Lethal trifecta* (short form): private data + untrusted input + a way to send data out, in one agent, lets an injection leak data. Full treatment 4.3. ⟵ N3
- **N97** · 3.1 · OpenClaw (always-on personal agent on Pi: gateway, chat channels, heartbeat, SOUL.md, ClawHub skills) and its 2026 incidents; Hermes Agent (self-written skills, capped memory); Mercury Agent (budgets and permission scopes as config). ⟵ N80, N93, N95, N96
- **N98** · 3.1 · Compare harnesses on dimensions: loop and interfaces, context files, memory, skills, permissions, sandbox. ⟵ N97
- **N99** · 3.1 · Three control points: permission (allow / ask / deny per tool), a before-tool hook (inspect and block), an after-tool hook (redact, trim, audit log). ⟵ N58
- **N100** · 3.2 · Build vs buy: raw API + your loop → agent SDK → open harness → hosted agents. ⟵ N55
- **N101** · 3.2 · OpenAI Agents SDK: `Agent`, `Runner`, function tools, *handoffs* (pass the conversation to another agent), *guardrails* (checks that can stop a run), *sessions* (stored history), tracing; any model via Chat Completions. ⟵ N19, N100
- **N102** · 3.2 · Approvals pause a run; its state (history, pending call) is serialised and resumed later, because the model remembers nothing. ⟵ N14, N101
- **N103** · 3.2 · Claude Agent SDK: Claude Code's harness as a library. Hosted agents (OpenAI Agents API, Claude Managed Agents) run the loop and sandbox for you. *ZDR* (zero data retention) and *data residency* decide whether that is allowed under PDPL. ⟵ N4, N56, N100
- **N104** · 3.3 · Long tasks fail by doing too much at once, declaring victory early, and forgetting; the fix is state in the repository (feature list, progress file, git). ⟵ N61
- **N105** · 3.3 · The Ralph loop: an outer loop that starts the agent with a fresh context each iteration, keeps state outside the model, uses tests as back-pressure, does one thing per iteration. ⟵ N50, N104
- **N106** · 3.3 · Tests push back only if the agent cannot edit them (rule in AGENTS.md *and* in the tool); review the diff. ⟵ N79, N92
- **N107** · 3.3 · Four exits (success, iteration budget, no progress, cost/time); a commit per iteration; maker and checker. ⟵ N53, N105
- **N108** · 3.4 · Eval vocabulary: task, trial, grader, transcript, outcome; grade outcomes, not claims. ⟵ N9, N52
- **N109** · 3.4 · Graders: code (strings, tool calls, database), *LLM judge* (a model scoring with a rubric; validate it), human review. ⟵ N108
- **N110** · 3.4 · pass@k (at least one of k trials passes) vs pass^k (all k pass, ≈ p^k). ⟵ N9, N108
- **N111** · 3.4 · Error analysis: read failures, write root causes, group, fix the largest group with one change, re-run. ⟵ N108
- **N112** · 3.4 · *OpenTelemetry*: the open standard for traces; a trace is a tree of *spans* (timed operations with attributes); the GenAI conventions name `gen_ai.*` attributes for model and tool calls. ⟵ N52

## Day 4 · Graph engineering and production

- **N113** · 4.1 · A *graph* makes a workflow explicit: nodes (steps), edges (fixed or conditional transitions), shared typed state. ⟵ N48
- **N114** · 4.1 · Five workflow patterns are graph shapes: prompt chain, routing, parallelisation, orchestrator–workers, evaluator–optimiser; plus human gates and checkpoints. ⟵ N113
- **N115** · 4.1 · LangGraph: `StateGraph` over a `TypedDict`, nodes return only changed keys, *reducers* merge them, conditional edges route, `compile(checkpointer=…)` saves state after every node. ⟵ N113
- **N116** · 4.1 · *Durable execution*: `interrupt()` pauses at a human gate; `Command(resume=…)` continues on the same `thread_id` hours or a crash later; the state history is the audit trail. ⟵ N102, N115
- **N117** · 4.1 · Loop or graph: unknown steps with tests as oracle → loop; known, regulated steps with decisions and audit → graph. ⟵ N105, N113
- **N118** · 4.2 · Many agents help on breadth (parallel reads in isolated contexts, at ~15× tokens) and hurt on coupled decisions; parallelise reads, keep writes in one place. ⟵ N63, N48
- **N119** · 4.2 · Agents as tools (fresh context per sub-question, the orchestrator keeps the thread) vs handoffs (the specialist takes over). ⟵ N101, N118
- **N120** · 4.2 · *A2A*: agents as services: an agent card at a well-known URL, messages in, tasks (with states and artifacts) out; MCP connects agents to tools, A2A connects agents to agents. ⟵ N83, N119
- **N121** · 4.3 · OWASP (an open security community known for its web Top 10) published the Top 10 for Agentic Applications (ASI01–ASI10). ⟵ N3
- **N122** · 4.3 · The lethal trifecta in full; defence = cut a leg. ⟵ N96
- **N123** · 4.3 · Defence layers, weakest to strongest: trust labels, input classifiers, least privilege, egress control, human approval, quarantine (a tool-less model turns untrusted text into typed facts: dual LLM, CaMeL). ⟵ N99, N122
- **N124** · 4.3 · Prove it with a red-team suite scored on two numbers: attack success (must be 0) and benign success (must stay high). KSA context: PDPL, SDAIA's AI Ethics Principles, NCA (National Cybersecurity Authority) guidance. ⟵ N108, N123
- **N125** · 4.4 · A service around the loop: FastAPI (a Python web framework with Pydantic validation), API keys, sessions scoped to a key, token budgets with 429, write tools queued for an officer, contract tests. ⟵ N26, N53
- **N126** · 4.4 · Fallbacks, caching, routing (small model for easy turns), versioning prompt + tools + model + graph as one release. ⟵ N22, N64
- **N127** · 4.4 · Production traces: export spans with OTLP to a backend; alert on errors, tokens, steps, guard blocks, budget 429s. ⟵ N112
- **N128** · 4.4 · Release gate: contract tests → smoke eval in CI → full suite → rollout (shadow traffic: run the new version on copies of real requests without showing its answers; canary: a small share of users). ⟵ N108, N125
- **N129** · 4.4 · Capstone: three tracks and the minimum (3+ tools, one over MCP; context engineering; an approval and budgets; 15+ eval tasks on two configurations; a trifecta analysis with one attack and one defence). ⟵ all

# Pass 1 log: gaps found in the original decks

Each row: original slide → what a reader with only the audience line (and the nuggets so far) was missing → the fix.

**Day 1 (rewritten end to end)**

| Original slide | Missing background | Fix |
|---|---|---|
| Course at a glance; the ladder | "API shapes", "context", "harness", "MCP", "LangGraph", "Ralph" all undefined on slide 2 | Opening describes each day in plain words (N1); the ladder moves to the end of Day 1 (N57), once prompt, loop and harness are defined |
| How we work: "you have called an LLM API before" | Contradicts the audience | Assumptions rewritten to the audience line |
| Case study: "prompt injections", "PDPL" | Neither defined | N3, N4 on the case-study slide |
| Rafeeq's tools (opening) | Tool calling unknown at this point | Moved to 1.3 (N37), after tool calling is explained |
| Setup: "all three API shapes" | Shapes unknown | Setup slide speaks only of the key and the check notebook |
| S1 poll "which is an agent"; definition; autonomy spectrum | Needs tools and loops (not yet taught) | Moved to 1.4, where tools are known (N48, N49) |
| S1 "three API shapes" table | No idea of an API, messages, roles, tool calls | 1.1 now starts with seven background slides: what a model is (N6–N9), LLM vs ChatGPT (N10–N12), chat format and statelessness (N13–N15), limits (N16), what an API is and who serves it (N17–N19); the three-shape table moves to 1.2 (N29) |
| S1 "one client, many providers" | base_url only makes sense once providers are known | After N18–N19 |
| S1 benchmarks: BFCL, τ²-bench, Terminal-Bench, "harness" | Function calling and harness unknown | Tool-calling preview (N23); harness phrased as "the agent software around the model" |
| S1 tokens slide (last) | Tokens are used from the first slide on | Tokens taught second (N7); the cost slide stays with the response fields |
| S2 items slide: `function_call`, `reasoning` | Tool calls and reasoning models unknown | Reasoning models explained (N31); function_call items shown as "tool calls, next session" |
| S2 structured outputs: "constrained decoding" | Needs generation-by-token (N8) | Now explained from N8 (N27); JSON Schema/Pydantic slide added (N26) |
| S2 built-in tools; timeline (Agents API, Agent Builder, Agents SDK); prompts as specs | Tools, agents and SDKs unknown | Hosted tools → 1.3 (N44); timeline removed (each item now sits where it is used: Assistants in N33, Agents API on Day 3); prompts as specs → 1.3 (N45) |
| S2 Decisions API: probabilities | Needs the token distribution | Points back to N6 |
| S3 Messages basics first | Tool calling never explained in general | 1.3 opens with tool calling in the shape you know (Chat Completions), then the Messages shape (N35–N39) |
| S3 Messages features: cache_control, memory tool, compaction, MCP connector | Day 2 concepts | Kept as a one-line forward map with plain descriptions |
| Lab 1.3 uses `K.run_agent` | Loop not yet taught | N47 states it on the lab slide |
| S4 frameworks: "compaction, hooks, guardrails, handoffs" | Undefined | Each gets a plain one-line description; harness and coding agents defined (N55, N56) |
| Quizzes with three options, answers on C, A, B, C | Style guide asks four options, spread A–D | Four options each; answers spread across the deck |

**Day 2**

| Original slide | Missing background | Fix |
|---|---|---|
| Yesterday: "hooks: after_tool, permission, approve" | Hooks undefined | Defined (N58) |
| Every step resends: "Manus" | Who is Manus | "an agent product" (N60) |
| Caching | What is cached and why it is cheaper | Background slide on what the provider caches (N64) |
| Minimisation: PDPL | Defined on Day 1 now (N4) | Points back |
| Pipeline vs tool: "RAG" | Never expanded | RAG and chunks defined first (N67) |
| Retrievers: BM25, embeddings, RRF, recall@1 | No IR background | Background slide on keyword vs vector search (N69, N70); recall@k defined (N73) |
| Memory: Hermes caps | Hermes appears on Day 3 | Introduced inline as "an open-source agent (Day 3)" |
| AGENTS.md: Codex, Cursor, Copilot, Gemini CLI | Coding agents | Defined on Day 1 (N56) |
| Skills: ClawHub, OpenClaw | Day 3 | Inline gloss |
| MCP: JSON-RPC, stdio, PEP 723 | Not in the audience line | Background on JSON-RPC (N82) and PEP 723 (N86) |
| MCP supply chain | Prompt injection | Defined on Day 1 (N3) |

**Day 3**

| Original slide | Missing background | Fix |
|---|---|---|
| Model + harness: LangChain | Who | Gloss (N91) |
| OpenClaw: CVE, "lethal trifecta (Day 4)" | Both undefined | CVE (N95) and a short trifecta definition (N96) |
| Agents SDK: handoffs, guardrails, sessions | Undefined | One line each (N101) |
| Hosted agents: ZDR | Undefined | Defined (N103) |
| Evals: "LLM judge", "stochastic" | Judge undefined; stochastic from N9 | N109 |
| Traces: OpenTelemetry, spans | Undefined | Background on spans (N112) |

**Day 4**

| Original slide | Missing background | Fix |
|---|---|---|
| LangGraph | Who makes it; what a graph is in this sense | N113, N115 |
| Durable execution | Term | N116 |
| OWASP | Who | N121 |
| Layers: dual LLM, CaMeL | Undefined | Plain description (N123) |
| NCA | Acronym | Expanded (N124) |
| Service: FastAPI; release gate: shadow, canary | Terms | N125, N128 |

**Day 5 and the labs**

| Item | Finding | Fix |
|---|---|---|
| Day 5 (capstone) | Every term it uses (ladder, trifecta, pass^k, MCP, approvals, budgets) is taught by Day 4 | No change |
| Lab intros 1.1–1.3 | Old session titles; Lab 1.3 used `K.run_agent` without saying what it is | Intros recap the session's nuggets (N6–N14, N29–N30, N35) and N47 is stated before the exercise |
| Labs, Day 2–4 | Session titles unchanged; each lab uses only nuggets taught in its session or earlier | No change |
