# %% [markdown]
# # Lab 2.3 · Remember and specialise
#
# Day 2 · Session 3 (Memory, skills and context files) · 25 min. A conversation ends; the context window is gone.
# What should Rafeeq carry into the next one, and what should it load only when needed? In this lab you:
# - implement the **memory-tool contract** (the command set of Anthropic's `memory` tool) over a capped folder;
# - make Rafeeq remember a preference across two separate sessions, and refuse to remember personal data;
# - load **AGENTS.md** as standing context;
# - add **Agent Skills** (`SKILL.md`) with progressive disclosure, and measure the tokens it saves.
#
# **Tasks**
# - [ ] 1 · The memory tool contract (8')
# - [ ] 2 · Remember across sessions (5')
# - [ ] 3 · AGENTS.md and skills with progressive disclosure (9')
# - [ ] 4 · Did the agent load the right skill? (3')
# - [ ] ✱ Stretch · A capped memory, Hermes-style

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/agentic-ai.git"
    if not Path("/content/agentic-ai").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/agentic-ai"], check=True)
    os.chdir("/content/agentic-ai/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("agentkit.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import agentkit as K
print("provider:", K.PROVIDER, "| model:", K.MODEL, "| cache:", K.USE_CACHE)

# %%
import json
import re
from typing import Literal

import pandas as pd

import madar

WORK = K.lab_dir("D2_3", reset=True)
m = madar.Madar(WORK)
READ = m.tools(m.READ_TOOLS)

# %% [markdown]
# ## 1 · The memory tool contract (8')
#
# Anthropic's `memory` tool (type `memory_20250818`) is a **contract**: the model sends commands, and **your code**
# stores the files wherever you like. The model only sees paths under `/memories`. The commands:
#
# | command | arguments | does |
# |---|---|---|
# | `view` | `path` | list a directory, or show a file with line numbers |
# | `create` | `path`, `file_text` | create or overwrite a file |
# | `str_replace` | `path`, `old_str`, `new_str` | replace one exact, unique occurrence |
# | `insert` | `path`, `insert_line`, `insert_text` | insert text after a line (0 = top) |
# | `delete` | `path` | delete a file or directory |
# | `rename` | `old_path`, `new_path` | rename or move |
#
# Two rules make it safe: every path must resolve **inside** the memory root (no `../` escapes), and the store has a
# **size cap** so memory can't grow without bound.
#
# ✏️ **Exercise** · Complete `Memory._resolve` (map `/memories/...` to a real path under `self.root`; raise
# `ValueError` for anything else, including traversal) and the `create` and `str_replace` commands with the
# `MAX_FILE` / `MAX_TOTAL` caps. The other commands are written for you.

# %%
class Memory:
    MAX_FILE = 1_500    # characters per file
    MAX_TOTAL = 4_000   # characters across the store

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        # BEGIN SOLUTION
        if not path or not (path == "/memories" or path.startswith("/memories/")):
            raise ValueError(f"Paths must start with /memories, got {path!r}")
        real = (self.root / path.removeprefix("/memories").lstrip("/")).resolve()
        if real != self.root and self.root not in real.parents:
            raise ValueError(f"Path escapes the memory directory: {path!r}")
        return real
        # END SOLUTION

    def _total(self, excluding: Path | None = None) -> int:
        return sum(len(p.read_text(encoding="utf-8")) for p in self.root.rglob("*") if p.is_file() and p != excluding)

    def view(self, path: str = "/memories") -> str:
        p = self._resolve(path)
        if p.is_dir():
            files = sorted(str(f.relative_to(self.root)).replace("\\", "/") for f in p.rglob("*") if f.is_file())
            return "\n".join(f"/memories/{f}" for f in files) or "(empty)"
        if not p.exists():
            raise ValueError(f"{path} does not exist")
        return "\n".join(f"{i:4d}\t{line}" for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1))

    def create(self, path: str, file_text: str) -> str:
        # BEGIN SOLUTION
        p = self._resolve(path)
        if len(file_text) > self.MAX_FILE:
            raise ValueError(f"File too long ({len(file_text)} > {self.MAX_FILE} chars). Keep only what matters.")
        if self._total(excluding=p) + len(file_text) > self.MAX_TOTAL:
            raise ValueError("Memory is full. Delete or shorten old memories first.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text, encoding="utf-8")
        return f"Created {path}"
        # END SOLUTION

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        # BEGIN SOLUTION
        p = self._resolve(path)
        text = p.read_text(encoding="utf-8")
        n = text.count(old_str)
        if n != 1:
            raise ValueError(f"old_str must occur exactly once in {path}; found {n} times.")
        return self.create(path, text.replace(old_str, new_str)).replace("Created", "Edited")
        # END SOLUTION

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        p = self._resolve(path)
        lines = p.read_text(encoding="utf-8").splitlines()
        lines.insert(insert_line, insert_text.rstrip("\n"))
        return self.create(path, "\n".join(lines)).replace("Created", "Edited")

    def delete(self, path: str) -> str:
        p = self._resolve(path)
        if p == self.root:
            raise ValueError("Refusing to delete the whole memory directory")
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
        else:
            p.unlink()
        return f"Deleted {path}"

    def rename(self, old_path: str, new_path: str) -> str:
        src, dst = self._resolve(old_path), self._resolve(new_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return f"Renamed {old_path} → {new_path}"

    def tool(self):
        """The contract as one tool function for run_agent (any API shape)."""
        mem = self

        def memory(command: Literal["view", "create", "str_replace", "insert", "delete", "rename"],
                   path: str | None = None, file_text: str | None = None, old_str: str | None = None,
                   new_str: str | None = None, insert_line: int | None = None, insert_text: str | None = None,
                   old_path: str | None = None, new_path: str | None = None) -> str:
            """Your long-term memory: files under /memories that persist between conversations.

            Args:
                command: view, create, str_replace, insert, delete or rename.
                path: a path under /memories, e.g. /memories/preferences.md.
                file_text: full content for create.
                old_str: exact text to replace (str_replace).
                new_str: replacement text (str_replace).
                insert_line: line number to insert after (insert; 0 = top).
                insert_text: text to insert (insert).
                old_path: source path (rename).
                new_path: destination path (rename).
            """
            args = {k: v for k, v in dict(path=path, file_text=file_text, old_str=old_str, new_str=new_str,
                                           insert_line=insert_line, insert_text=insert_text, old_path=old_path,
                                           new_path=new_path).items() if v is not None}
            try:
                return getattr(mem, command)(**args)
            except (ValueError, OSError, TypeError) as e:
                return json.dumps({"error": str(e)})

        return memory


mem = Memory(WORK / "memories")
mem.create("/memories/preferences.md", "- Prefers answers in English\n")
mem.str_replace("/memories/preferences.md", "English", "English, short")
print(mem.view(), "\n" + mem.view("/memories/preferences.md"))
for bad in ["/memories/../../secrets.txt", "/etc/passwd", "memories/x.md"]:
    try:
        mem.view(bad)
        K.check(False, f"blocked {bad}")
    except ValueError:
        K.check(True, f"blocked {bad}")
K.check("error" in mem.tool()("create", path="/memories/big.md", file_text="x" * 5000), "the size cap holds")

# %% [markdown]
# ## 2 · Remember across sessions (5')
#
# Two **separate** runs, no shared history: only `/memories` connects them. The system prompt says what memory is
# for and, just as important, what it must never hold.

# %%
MEMORY_RULES = """
## Memory
You have a memory tool. At the start of every conversation, view /memories. Save durable preferences
(language, level of detail, preferred contact time) in /memories/preferences.md.
Never store national ID digits, mobile numbers, permit details or anything else from a permit file (PDPL)."""
SYS = madar.SYSTEM_PROMPT + MEMORY_RULES
mem = Memory(WORK / "memories_sara")
memory = mem.tool()

s1 = K.run_agent("Hi, I'm Sara. From now on please always answer me in English, even when I write in Arabic, and "
                 "keep it short. My permit is REN-10482, ID ending 4821, mobile 0501234821.",
                 [memory, *READ], system=SYS, max_steps=8)
print("\nmemory after session 1:\n" + mem.view())

# %%
s2 = K.run_agent("كم رسوم لوحة مضيئة مساحتها 3 أمتار مربعة؟", [memory, *READ], system=SYS, max_steps=8)
stored = "\n".join(p.read_text(encoding="utf-8") for p in (WORK / "memories_sara").rglob("*") if p.is_file())
K.check(any(n == "memory" for n, _ in s2.tool_calls), "session 2 read its memory")
K.check(not re.search("[؀-ۿ]", s2.final), "session 2 answered in English, as remembered")
K.check("4821" not in stored and "0501234821" not in stored, "no ID digits or mobile in memory")

# %% [markdown]
# ## 3 · AGENTS.md and skills with progressive disclosure (9')
#
# Two kinds of standing context:
# - **AGENTS.md**: a short, always-loaded file of project conventions (an open format used by many coding agents).
#   Here it's `data/AGENTS.md`.
# - **Skills**: folders with a `SKILL.md` (YAML front matter `name` + `description`, then instructions). Only the
#   names and descriptions sit in the context; the agent **loads** a skill's body when a task needs it. That is
#   *progressive disclosure*.

# %%
AGENTS_MD = (K.DATA / "AGENTS.md").read_text(encoding="utf-8")
SKILLS_DIR = K.DATA / "skills"
print(AGENTS_MD[:400], "…\n")
print([p.parent.name for p in sorted(SKILLS_DIR.glob("*/SKILL.md"))])

# %% [markdown]
# ✏️ **Exercise** · Write:
# - `read_skill(path)` → `(meta, body)`: parse the front matter between the first two `---` lines into a dict
#   (`key: value` per line) and return the rest as the body;
# - `skill_index()` → a Markdown list `- **name**: description` for every skill;
# - `load_skill(name)` → a tool that returns the body of one skill, or an error that lists the available names.

# %%
# BEGIN SOLUTION
def read_skill(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    _, head, body = raw.split("---", 2)
    meta = dict(line.split(": ", 1) for line in head.strip().splitlines() if ": " in line)
    return meta, body.strip()


SKILLS = {read_skill(p)[0]["name"]: p for p in sorted(SKILLS_DIR.glob("*/SKILL.md"))}


def skill_index() -> str:
    return "\n".join(f"- **{name}**: {read_skill(p)[0]['description']}" for name, p in SKILLS.items())


def load_skill(name: str) -> str:
    """Load the step-by-step instructions of one skill before doing that kind of task.

    Args:
        name: the skill name from the list in your instructions, e.g. "refund-request".
    """
    if name not in SKILLS:
        return json.dumps({"error": f"No skill {name!r}. Available: {sorted(SKILLS)}"})
    return read_skill(SKILLS[name])[1]
# END SOLUTION


print(skill_index())
K.check(len(SKILLS) == 4 and "verify" in load_skill("refund-request").lower(), "skills parsed and loadable")

# %% [markdown]
# Compare the two ways to give Rafeeq its procedures: **inline** (every skill body in the system prompt) or
# **progressive** (the index in the prompt, bodies on demand).

# %%
SYS_INLINE = "\n\n".join([madar.SYSTEM_PROMPT, AGENTS_MD, "# Procedures"] + [read_skill(p)[1] for p in SKILLS.values()])
SYS_PROGRESSIVE = "\n\n".join([madar.SYSTEM_PROMPT, AGENTS_MD,
                               "# Skills\nBefore a multi-step task, call load_skill with the matching name:\n" + skill_index()])
print(f"inline ≈ {K.count_tokens(SYS_INLINE)} tokens | progressive ≈ {K.count_tokens(SYS_PROGRESSIVE)} tokens per call")

# %% [markdown]
# With four skills the saving is modest; with forty (a typical coding agent), inlining stops being an option.

# %% [markdown]
# ## 4 · Did the agent load the right skill? (3')
#
# Khalid was charged twice. The right procedure is `refund-request`: verify, state the amount, then create.

# %%
KHALID = ("I was charged twice for my sign permit SGN-40388. Please refund the second payment. "
          "ID ends 2277, mobile 0505502277. Yes, go ahead and create the refund.")
r = K.run_agent(KHALID, [load_skill, *m.tools(m.READ_TOOLS + ("create_refund",))], system=SYS_PROGRESSIVE, max_steps=8)
loaded = [a.get("name") for n, a in r.tool_calls if n == "load_skill"]
refunds = m.sql("SELECT refund_id, amount_sar, reason, status FROM refunds WHERE permit_id = 'SGN-40388'")
print("skills loaded:", loaded, "| refunds:", refunds)
K.check("refund-request" in loaded, "the agent loaded the refund-request skill")
K.check(len(refunds) == 1 and refunds[0]["reason"] == "duplicate_payment", "one duplicate-payment refund was created")

# %% [markdown]
# ## ✱ Stretch · A capped memory, Hermes-style
#
# Hermes Agent keeps two small files, `MEMORY.md` (about 2,200 characters) and `USER.md` (about 1,375), and makes the
# agent **curate** them: when a file is full, the agent must merge or drop entries instead of appending. Change
# `MEMORY_RULES` and the caps to that design, then run five sessions that each add a preference. Inspect what the
# agent chose to keep.

# %%
# BEGIN SOLUTION
class CappedMemory(Memory):
    MAX_FILE = 1_375
    MAX_TOTAL = 1_375


cap = CappedMemory(WORK / "memories_capped")
rules = MEMORY_RULES + "\nUse only /memories/USER.md. It is capped at 1,375 characters: when an edit fails because " \
                       "it is full, rewrite the file shorter (merge or drop old entries) and try again."
for note in ["I prefer English.", "Call me after 4 pm.", "Keep answers under 3 sentences.",
             "I'm an engineer; technical terms are fine.", "Always mention the policy ID."][: K.pick(5, 2)]:
    K.run_agent(f"Please remember: {note}", [cap.tool()], system=madar.SYSTEM_PROMPT + rules, max_steps=6, verbose=False)
print(cap.view("/memories/USER.md") if (cap.root / "USER.md").exists() else cap.view())
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Why does the memory contract run on **your** side, even when the model and the tool definition come from Anthropic?
# 2. What would happen to Madar if Rafeeq stored national IDs in memory "to save the user time"?
# 3. When does a skill belong in AGENTS.md instead, and vice versa?
# 4. Progressive disclosure saves tokens. What does it cost?
